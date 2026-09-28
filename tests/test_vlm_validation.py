import sys
import os

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from genai.vlm import validate_vlm_response


# Verified information from YOLO + Navigation
target_object = "chair"
target_position = "right"
target_direction = "left"


# -------------------------------------------------
# TEST 1: Correct VLM response
# -------------------------------------------------

correct_response = (
    "A chair is on the right. Move left carefully."
)

print("\n==============================")
print("TEST 1: CORRECT RESPONSE")
print("==============================")

result = validate_vlm_response(
    correct_response,
    target_object,
    target_position,
    target_direction
)

print("Expected: True")
print("Actual  :", result)


# -------------------------------------------------
# TEST 2: Wrong direction
# -------------------------------------------------

wrong_direction_response = (
    "A chair is on the right. Move right carefully."
)

print("\n==============================")
print("TEST 2: WRONG DIRECTION")
print("==============================")

result = validate_vlm_response(
    wrong_direction_response,
    target_object,
    target_position,
    target_direction
)

print("Expected: False")
print("Actual  :", result)


# -------------------------------------------------
# TEST 3: Wrong position
# -------------------------------------------------

wrong_position_response = (
    "A chair is on the left. Move left carefully."
)

print("\n==============================")
print("TEST 3: WRONG POSITION")
print("==============================")

result = validate_vlm_response(
    wrong_position_response,
    target_object,
    target_position,
    target_direction
)

print("Expected: False")
print("Actual  :", result)


# -------------------------------------------------
# TEST 4: Wrong object
# -------------------------------------------------

wrong_object_response = (
    "A table is on the right. Move left carefully."
)

print("\n==============================")
print("TEST 4: WRONG OBJECT")
print("==============================")

result = validate_vlm_response(
    wrong_object_response,
    target_object,
    target_position,
    target_direction
)

print("Expected: False")
print("Actual  :", result)