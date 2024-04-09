import cv2
import pytesseract


class LinEqSys:
    def __init__(self):
        ...

    def add_equation(self, new_equation: str):
        ...

    def read_from_image(self, image_path: str):
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text_result = pytesseract.image_to_string(img, timeout=2)
            print(text_result)

        except FileNotFoundError:
            print("File not found")
            return


if __name__ == "__main__":
    test_sys = LinEqSys()
    test_sys.read_from_image("test_image.png")
