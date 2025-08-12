# solve_recaptcha.py  (patched)
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import requests

from PIL import Image
from io import BytesIO
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from models.YOLO_Classification import predict
from models.YOLO_Segment import predict as predict_segment
import time
import csv
from datetime import datetime
from IP import vpn
from selenium.webdriver import ActionChains
from pynput.mouse import Button, Controller
import random
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import WebDriverException, StaleElementReferenceException, NoSuchElementException
import traceback
import re
from typing import Optional

# ---------------------
# Configuration / Flags
# ---------------------
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo"
THRESHOLD = 0.2
USE_TOP_N_STRATEGY = False
N = 3
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic']
MODEL = "YOLO" # "YOLO"
TYPE1 = True #one time image selection
TYPE2 = True #segmentation problem
TYPE3 = True #dynamic captcha

ENABLE_LOGS = True
ENABLE_VPN = False        # <-- keep VPN disabled by default (your request)
ENABLE_MOUSE_MOVEMENT = True
ENABLE_NATURAL_MOUSE_MOVEMENT = True

# IMPORTANT: set to False by default to avoid invalid profile path errors on server
ENABLE_COOKIES = False
PATH_TO_FIREFOX_PROFILE = ''  # if you want to use a profile put absolute server path here

# ---------------------
# Utility & setup
# ---------------------
# suppress tensorflow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

# Check if data dir is present
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

# ---------------------
# Model loader
# ---------------------
def getFirstModel():
    model_path = "models/Base_Line/first_model.h5"
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=["accuracy"])
    return model

# ---------------------
# Helpers (safe_find, bezier mouse)
# ---------------------
def safe_find(container, by, value, timeout=5, retries=3, delay=0.5):
    """
    Try to find an element and recover from StaleElementReferenceException or temporary absence.
    container can be driver or a WebElement.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            # WebDriverWait works with driver; if container is an element, call find_element directly
            if hasattr(container, 'find_element'):
                # this will raise if not found
                el = container.find_element(by, value)
                return el
            else:
                # fallback: assume container is driver-like
                el = container.find_element(by, value)
                return el
        except StaleElementReferenceException as e:
            last_exc = e
            time.sleep(delay)
        except NoSuchElementException as e:
            last_exc = e
            time.sleep(delay)
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    # final try (let exception bubble)
    return container.find_element(by, value)

def generate_bezier_curve(p0, p1, p2, p3, num_points=100):
    """ Generate points along a Bezier curve using four control points. """
    curve = []
    for t in np.linspace(0, 1, num_points):
        point = (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3
        curve.append(point)
    return curve

def move_mouse_in_curve(mouse, start_pos, end_pos):
    """ Move the mouse in a curve from start_pos to end_pos. """
    ctrl1 = start_pos + np.random.rand(2) * 100
    ctrl2 = end_pos + np.random.rand(2) * 100
    curve = generate_bezier_curve(np.array(start_pos), ctrl1, ctrl2, np.array(end_pos))
    for point in curve:
        try:
            mouse.position = (int(point[0]), int(point[1]))
        except Exception:
            # ignore occasional failures to set hardware mouse position
            pass
        time.sleep(random.uniform(0.003, 0.01))

mouse = Controller()

# ---------------------
# Prediction helpers
# ---------------------
def predict_tile_local(tile, model):    
    i = img_to_array(tile)
    to_predict = i.reshape((-1,224,224,3))
    prediction = model.predict(to_predict)
    return [ prediction, CLASSES[np.argmax(prediction)], np.argmax(prediction)  ]

# ---------------------
# Core UI interaction
# ---------------------
def click_element(driver, element, offset=True):
    # Execute JavaScript to get the absolute position of the element
    if ENABLE_MOUSE_MOVEMENT:
        try:
            x = driver.execute_script('return arguments[0].getBoundingClientRect().left + window.pageXOffset', element)
            y = driver.execute_script('return arguments[0].getBoundingClientRect().top + window.pageYOffset', element)
        except Exception:
            # fallback: try using element location
            loc = element.location
            x, y = loc.get('x', 0), loc.get('y', 0)

        # Add offsets used previously
        if offset:
            x += 110
            y += 220
        else:
            x += 30
            y += 440

        if ENABLE_NATURAL_MOUSE_MOVEMENT:
            start_pos = np.array(mouse.position)
            end_pos = np.array([x, y])
            move_mouse_in_curve(mouse, start_pos, end_pos)
        else:
            try:
                mouse.position = (int(x), int(y))
            except Exception:
                pass

        sleep(0.1)
        try:
            mouse.click(Button.left, 1)
        except Exception:
            # fallback to element.click if hardware click fails
            try:
                element.click()
            except Exception:
                pass
        sleep(0.2)
    else:
        try:
            element.click()
        except Exception:
            pass

# ---------------------
# Tile processing
# ---------------------
COUNT = 0

def process_tile(i, model, captcha_object, class_index, driver):
    global COUNT
    print("processing tile with class index ", str(class_index))

    xpath = "//td[contains(@tabindex, '" + str(i+4)+ "')]"
    try:
        matched_tile = safe_find(driver, By.XPATH, xpath, timeout=5, retries=4)
    except Exception as e:
        print("Could not find matched tile:", e)
        return False

    filename = f"tile_{COUNT}.jpg"
    try:
        matched_tile.screenshot(os.path.join(data_dir, filename))
    except Exception as e:
        print("Failed to screenshot tile:", e)
        sleep(0.5)
        try:
            matched_tile = safe_find(driver, By.XPATH, xpath, retries=2)
            matched_tile.screenshot(os.path.join(data_dir, filename))
        except Exception as e2:
            print("Second screenshot attempt failed:", e2)
            return False

    try:
        img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
        img = img.resize(size=(224,224))
    except Exception as e:
        print("Failed to open/resize tile image:", e)
        return False

    try:
        if MODEL == "YOLO":
            result = predict.predict_tile(os.path.join(data_dir, filename))
            current_object_probability = result[0][class_index]
            object_name = YOLO_CLASSES[result[2]]
        else:
            result = predict_tile_local(img, model)
            current_object_probability = result[0][0][class_index]
            object_name = CLASSES[result[2]]
    except Exception as e:
        print("Prediction failed:", e)
        return False

    #rename image
    try:
        os.rename(os.path.join(data_dir, filename), os.path.join(data_dir, object_name + "_" + filename))
    except Exception:
        pass

    print(f"{COUNT}: The AI predicted tile to be {object_name} and probability is {current_object_probability}")

    COUNT += 1
    if USE_TOP_N_STRATEGY:
        top_n_indices = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:N]
        if class_index in top_n_indices:
            click_element(driver, matched_tile)
            return True
    else:
        if current_object_probability > THRESHOLD:
            print(current_object_probability, " > ", THRESHOLD)
            click_element(driver, matched_tile)
            return True

    return False

# ---------------------
# Solve type2 (segmentation)
# ---------------------
def solve_type2(driver):
    save_path = "temp"
    os.makedirs(save_path, exist_ok=True)
    xpath_image = "/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[1]/td[1]/div/div[1]/img"
    xpath_text = "/html/body/div/div/div[2]/div[1]/div[1]/div/strong"

    try:
        captcha_text_el = safe_find(driver, By.XPATH, xpath_text, timeout=5, retries=3)
        captcha_text = captcha_text_el.text
    except Exception as e:
        print("Could not read captcha text for type2:", e)
        return

    log("Type2", captcha_text)

    class_index = None
    for i in CLASSES:
        if i in captcha_text:
            class_index = CLASSES.index(i)
            break
    if class_index is None:
        print("Could not determine class index from captcha text:", captcha_text)
        return

    try:
        img = safe_find(driver, By.XPATH, xpath_image, timeout=5, retries=3)
        img_url = img.get_attribute("src")
        response = requests.get(img_url, stream=True, timeout=10)
    except Exception as e:
        print("Failed to download captcha image:", e)
        return

    if response.status_code == 200:
        timestamp = str(time.time())
        filename = f"image_{captcha_text}_{timestamp}.png"
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(response.content)
        success, grid = predict_segment.predict(class_index, os.path.join(save_path, filename))
        xpath_tiles = "/html/body/div/div/div[2]/div[2]/div/table/tbody"
        tiles_to_click = [(i+1, j+1) for i in range(4) for j in range(4) if grid[i][j] == 1]
        for i, j in tiles_to_click:
            try:
                tile = safe_find(driver, By.XPATH, xpath_tiles + f"/tr[{i}]/td[{j}]", retries=3)
                click_element(driver, tile)
                sleep(0.5)
            except Exception as e:
                print("Failed clicking tile:", e)
    try:
        click_element(driver, safe_find(driver, By.ID, "recaptcha-verify-button"))
    except Exception:
        try:
            driver.find_element(By.ID, "recaptcha-verify-button").click()
        except Exception:
            pass
    sleep(0.5)

# ---------------------
# Browser open + captcha entry
# ---------------------
def open_browser_with_captcha():
    if ENABLE_VPN:
        try:
            vpn.connect()
            print("VPN connected")
        except Exception as e:
            print("VPN connect failed:", e)
            if ENABLE_VPN:
                raise

    options = Options()
    driver = None
    try:
        if ENABLE_COOKIES and PATH_TO_FIREFOX_PROFILE:
            if os.path.exists(PATH_TO_FIREFOX_PROFILE):
                options.profile = PATH_TO_FIREFOX_PROFILE
                driver = webdriver.Firefox(options=options)
                print("init with cookies/profile")
            else:
                print("Requested Firefox profile not found. Starting with default profile.")
                driver = webdriver.Firefox()
        else:
            driver = webdriver.Firefox()
    except Exception as e:
        print("Failed to start Firefox webdriver:", e)
        if ENABLE_VPN:
            try:
                vpn.disconnect()
            except Exception:
                pass
        raise

    driver.maximize_window()

    for _ in range(10):
        try:
            driver.get(CAPTCHA_URL)
            WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
            sleep(1)
            recapcha_frame = safe_find(driver, By.XPATH, "//iframe[@title='reCAPTCHA']", timeout=5, retries=4)
            driver.switch_to.frame(recapcha_frame)
            checkbox = safe_find(driver, By.CLASS_NAME, "recaptcha-checkbox-border", timeout=5, retries=4)
            try:
                checkbox.click()
            except Exception:
                try:
                    driver.execute_script("arguments[0].click();", checkbox)
                except Exception:
                    pass
            sleep(4)
            driver.switch_to.default_content()
            WebDriverWait(driver, 3).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))
            print("opened the browser with the captcha")
            return driver
        except WebDriverException as e:
            print("An error occurred while opening captcha. Reloading the page and trying again.", e)
            try:
                driver.refresh()
            except Exception:
                pass
            sleep(1)
        except Exception as e:
            print("Unexpected error while trying to open captcha:", e)
            try:
                driver.refresh()
            except Exception:
                pass
            sleep(1)
    print("Failed to open the browser with the captcha after 10 attempts.")
    return None

# ---------------------
# Utility functions for captcha logic
# ---------------------
def get_class_index(captcha_object):
    for i in CLASSES:
        if i in captcha_object.text:
            if MODEL == "YOLO":
                return YOLO_CLASSES.index(i)
            else:
                return CLASSES.index(i)
    return None

def handle_dynamic_captcha(driver, model, captcha_object, class_index, to_check):
    if len(to_check) < 1:
        try:
            click_element(driver, safe_find(driver, By.ID, "recaptcha-verify-button"))
        except Exception:
            pass
        sleep(1)
        return

    while True:
        # iterate a *copy* because we may modify list inside loop
        for idx in list(to_check):
            try:
                if process_tile(idx, model, captcha_object, class_index, driver):
                    if idx not in to_check:
                        to_check.append(idx)
                else:
                    if idx in to_check:
                        to_check.remove(idx)
                sleep(2)
            except StaleElementReferenceException:
                print("Stale element while processing tile, will retry loop.")
                sleep(0.5)
                continue
            except Exception as e:
                print("Error processing tile in dynamic handler:", e)
                continue

        if len(to_check) < 1:
            try:
                click_element(driver, safe_find(driver, By.ID, "recaptcha-verify-button"))
            except Exception:
                pass
            sleep(2)
            try:
                error_message = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "rc-imageselect-error-select-more"))
                )
                if 'none' in error_message.get_attribute('style'):
                    print("The 'select more images' text did not appear.")
                else:
                    print("The 'select more images' text appeared.")
                    try:
                        driver.find_element(By.ID, "recaptcha-reload-button").click()
                    except Exception:
                        pass
            except Exception:
                print("The 'select more images' text did not appear or could not be read.")
            break

def captcha_is_solved(driver):
    sleep(1)
    try:
        driver.switch_to.default_content()
        iframe = safe_find(driver, By.XPATH, '/html/body/div[1]/form/fieldset/ul/li[5]/div/div/div/div/iframe', timeout=5, retries=3)
        driver.switch_to.frame(iframe)
        checkbox = safe_find(driver, By.XPATH, '//*[@id="recaptcha-anchor"]', timeout=5, retries=3)
        if checkbox.get_attribute('aria-checked') == 'true':
            print("captcha is solved")
            return True
        else:
            print("captcha is not solved yet")
            return False
    except Exception:
        return False
    finally:
        try:
            driver.switch_to.default_content()
            WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))
        except Exception:
            pass

# ---------------------
# Logging (robust)
# ---------------------
log_filename = None
session_folder = None

def log(captcha_type, captcha_object):
    global log_filename, session_folder

    if not ENABLE_LOGS:
        return

    if session_folder is None:
        highest_session_number = 0
        for dirname in os.listdir('.'):
            if dirname.startswith('Session'):
                m = re.search(r'(\d+)', dirname)
                if m:
                    try:
                        session_number = int(m.group(1))
                        highest_session_number = max(highest_session_number, session_number)
                    except Exception:
                        continue
        session_folder = f'Session{highest_session_number + 1:02}'
        os.makedirs(session_folder, exist_ok=True)
        save_global_variables()

    highest_log_number = 0
    if log_filename is None:
        for filename in os.listdir(session_folder):
            if filename.startswith('logs_'):
                m = re.search(r'logs_(\d+)', filename)
                if m:
                    try:
                        log_number = int(m.group(1))
                        highest_log_number = max(highest_log_number, log_number)
                    except Exception:
                        continue
        log_filename = os.path.join(session_folder, f'logs_{highest_log_number + 1:02}.csv')

    try:
        with open(log_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, captcha_type, captcha_object])
    except Exception as e:
        print("Failed to write log:", e)

def save_global_variables():
    try:
        with open(os.path.join(session_folder, 'global_variables.txt'), 'w') as file:
            file.write(f'CAPTCHA_URL = {CAPTCHA_URL}\n')
            file.write(f'THRESHOLD = {THRESHOLD}\n')
            file.write(f'CLASSES = {CLASSES}\n')
            file.write(f'YOLO_CLASSES = {YOLO_CLASSES}\n')
            file.write(f'MODEL = {MODEL}\n')
            file.write(f'TYPE1 = {TYPE1}\n')
            file.write(f'TYPE2 = {TYPE2}\n')
            file.write(f'TYPE3 = {TYPE3}\n')
            file.write(f'ENABLE_LOGS = {ENABLE_LOGS}\n')
            file.write(f'ENABLE_VPN = {ENABLE_VPN}\n')
            file.write(f'ENABLE_MOUSE_MOVEMENT = {ENABLE_MOUSE_MOVEMENT}\n')
            file.write(f'ENABLE_NATURAL_MOUSE_MOVEMENT = {ENABLE_NATURAL_MOUSE_MOVEMENT}\n')
            file.write(f'ENABLE_COOKIES = {ENABLE_COOKIES}\n')
            file.write(f'USE_TOP_N_STRATEGY = {USE_TOP_N_STRATEGY}\n')
            file.write(f'N = {N}\n')
    except Exception as e:
        print("Failed saving global variables:", e)

def reset_globals():
    global log_filename
    log_filename = None

# ---------------------
# Main solving loop
# ---------------------
def solve_classification_type(driver, model, dynamic_captcha):
    try:
        captcha_container = safe_find(driver, By.ID, 'rc-imageselect', timeout=5, retries=4)
        captcha_object = safe_find(captcha_container, By.TAG_NAME, 'strong', timeout=3, retries=3)
    except Exception as e:
        print("Failed to get captcha_object:", e)
        return

    class_index = get_class_index(captcha_object)
    if class_index is None:
        print("Could not identify class index from captcha_object text.")
        return

    try:
        if dynamic_captcha:
            log("dynamic", captcha_object.text)
        else:
            log("Type1", captcha_object.text)
    except Exception as e:
        print("Logging error (non-fatal):", e)

    to_check = []
    for i in range(9):
        try:
            if process_tile(i, model, captcha_object, class_index, driver):
                to_check.append(i)
        except Exception as e:
            print("Error while initial processing tile:", e)
            continue

    if dynamic_captcha:
        handle_dynamic_captcha(driver, model, captcha_object, class_index, to_check)
    else:
        try:
            click_element(driver, safe_find(driver, By.ID, "recaptcha-verify-button"))
        except Exception:
            try:
                driver.find_element(By.ID, "recaptcha-verify-button").click()
            except Exception:
                pass

# ---------------------
# Run loop
# ---------------------
def run():
    model = getFirstModel()
    try:
        driver = open_browser_with_captcha()
    except Exception as e:
        print("open_browser_with_captcha failed:", e)
        if ENABLE_VPN:
            try:
                vpn.disconnect()
            except Exception:
                pass
        return False

    while True:
        try:
            try:
                text = safe_find(driver, By.ID, 'rc-imageselect', timeout=5, retries=3).text
            except Exception:
                text = ""
            if "squares" in text and TYPE2:
                print("found a 4x4 segmentation problem")
                solve_type2(driver)
            elif "none" in text and TYPE3:
                print("found a 3x3 dynamic captcha")
                dynamic_captcha = True
                solve_classification_type(driver, model, dynamic_captcha)
            elif TYPE1:
                print("found a 3x3 one time selection captcha")
                dynamic_captcha = False
                solve_classification_type(driver, model, dynamic_captcha)
            else:
                try:
                    btn = driver.find_element(By.ID, "recaptcha-reload-button")
                    btn.click()
                except Exception:
                    pass
                continue

            if captcha_is_solved(driver):
                log("SOLVED", "captcha solved")
                try:
                    driver.close()
                except Exception:
                    pass
                if ENABLE_VPN:
                    try:
                        vpn.disconnect()
                    except Exception:
                        pass
                break

        except Exception as e:
            print("error occurred in main loop:", e)
            traceback.print_exc()
            if ENABLE_VPN:
                try:
                    vpn.disconnect()
                except Exception:
                    pass
            try:
                if captcha_is_solved(driver):
                    log("SOLVED", "captcha solved")
                    try:
                        driver.close()
                    except Exception:
                        pass
                    break
            except Exception:
                pass
            try:
                reload_btn = driver.find_element(By.ID, "recaptcha-reload-button")
                reload_btn.click()
            except Exception:
                pass
            continue

if __name__ == "__main__":
    run()

