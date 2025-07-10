import os
import cv2
import csv
import numpy as np


class StampDetector:
    def __init__(self, folder_read: str, folder_save: str, results_file: str):
        self.FOLDER_READ = folder_read
        self.FOLDER_SAVE = folder_save
        self.RESULTS_FILE = results_file
        self.results = []

    def find_stamp(self) -> None:
        for filename in os.listdir(self.FOLDER_READ):
            image = cv2.imread(self.FOLDER_READ + filename)

            if image is None:
                print(f'Failed to read: {filename}')
                continue

            transformed = self._transform_image(image)
            contours = self._draw_contour(transformed, image)
            info = self._check_side(contours, filename, image)
            self.results.append(info)

            is_saved = cv2.imwrite(self.FOLDER_SAVE + filename, image)

            if not is_saved:
                print(f'Failed to save: {filename}')

        self._save_info()

    def _transform_image(self, image: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Диапазоны синего в HSV
        lower_blue = np.array([60, 30, 50])
        upper_blue = np.array([180, 255, 255])

        result = cv2.inRange(hsv, lower_blue, upper_blue)

        # Добавляем блюр для избавления от шумов
        result = cv2.GaussianBlur(result, (9, 9), 3)

        # Преобразую в binary
        _, binary = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Дилатация для расширения / закрытия контура
        dilate = cv2.dilate(binary, kernel, iterations=2)

        return dilate

    def _draw_contour(self, dilate: np.ndarray, image: np.ndarray) -> list:

        # Находим контуры
        cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтруем по размеру, избавляемся от оставшейся загрязненности
        min_area = 500
        contours = [c for c in cnts if cv2.contourArea(c) > min_area]

        # Рисуем контур
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        return contours

    def _check_side(self, contours: list, image_name: str, image: np.ndarray) -> dict:
        image_center_x = image.shape[1] // 2

        if len(contours) > 1:
            result = {'Лист': image_name,
                      'Исполнитель': 'Да',
                      'Заказчик': 'Да'}

        elif not contours:
            result = {'Лист': image_name,
                      'Исполнитель': 'Нет',
                      'Заказчик': 'Нет'}
        else:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                X = int(M["m10"] / M["m00"])
            else:
                X = image_center_x

            result = {'Лист': image_name,
                      'Исполнитель': 'Да' if X < image_center_x else 'Нет',
                      'Заказчик': 'Да' if X > image_center_x else 'Нет'}
        return result

    def _save_info(self) -> None:
        with open(self.RESULTS_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(
                file, fieldnames=['Лист', 'Исполнитель', 'Заказчик'])
            writer.writeheader()
            writer.writerows(self.results)


if __name__ == '__main__':

    folder_read = './Images/'
    folder_save = './Images_processed/'
    final_list = './stamps_list.csv'

    StampDetector(folder_read, folder_save, final_list).find_stamp()