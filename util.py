import string
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

dict_char_to_int = {'А': '0', 'В': '1', 'Е': '2', 'К': '3', 'М': '4', 'Н': '5', 'О': '6', 'Р': '7', 'С': '8', 'Т': '9'}
dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}

def write_csv(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        for frame_nmr, cars in results.items():
            for car_id, car_details in cars.items():
                if 'car' in car_details and 'license_plate' in car_details and 'text' in car_details['license_plate']:
                    car_bbox = ' '.join(map(str, car_details['car']['bbox']))
                    lp_bbox = ' '.join(map(str, car_details['license_plate']['bbox']))
                    lp_text = car_details['license_plate']['text']
                    lp_text_score = car_details['license_plate']['text_score']
                    f.write(f'{frame_nmr},{car_id},[{car_bbox}],[{lp_bbox}],{car_details["license_plate"]["bbox_score"]},{lp_text},{lp_text_score}\n')

def license_complies_format(text):
    valid_format = 'XXX-9999'
    if len(text) != len(valid_format):
        return False
    for i, char in enumerate(text):
        if valid_format[i] == 'X' and (char.upper() not in string.ascii_uppercase and char not in dict_int_to_char):
            return False
        elif valid_format[i] == '9' and (char not in '0123456789' and char not in dict_char_to_int):
            return False
        elif valid_format[i] == '-' and char != '-':
            return False
    return True

def format_license(text):
    formatted = ''
    for i, char in enumerate(text):
        if char in dict_char_to_int and i < 3:  # First three characters are letters
            formatted += dict_char_to_int[char]
        elif char in dict_int_to_char and i > 3:  # Last four characters are digits
            formatted += dict_int_to_char[char]
        else:
            formatted += char
    return formatted

def read_license_plate(license_plate_crop):
    license_plate_crop = license_plate_crop.convert('L')
    text = pytesseract.image_to_string(license_plate_crop, lang='eng', config='--psm 6').upper().replace(' ', '')
    if license_complies_format(text):
        return format_license(text), 1.0
    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, _, _ = license_plate
    for track_id in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track_id
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return track_id
    return -1, -1, -1, -1, -1
