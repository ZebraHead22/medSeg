# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import design
import numpy as np
from PIL import Image
from mymodel import net
import SimpleITK as sitk
from PyQt5 import QtWidgets, QtCore
from ultralytics import YOLO
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageFile
from torchvision import transforms
# from radiomics import featureextractor

import copy
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import glob
import torch.nn.functional as F
from torchvision import transforms
import io

# Класс для перенаправления вывода в терминал
class StreamToTerminal(io.StringIO):
    def __init__(self, terminal_widget):
        super().__init__()
        self.terminal_widget = terminal_widget
        
    def write(self, message):
        super().write(message)
        self.terminal_widget.append(message)
        # Автоматическая прокрутка вниз
        scrollbar = self.terminal_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QtWidgets.QApplication.processEvents()
        
    def flush(self):
        pass

class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = os.getcwd()
        self.data = None
        
        # Отключаем автоматическое масштабирование в QLabel
        self.raw_image.setScaledContents(False)
        self.mask_l.setScaledContents(False)
        self.segm_l.setScaledContents(False)
        
        # Заменяем QLabel на QTextEdit для терминала
        self.term = QtWidgets.QTextEdit()
        self.term.setStyleSheet("""
            QTextEdit {
                border-radius: 5px;
                background-color: rgb(0, 0, 0);
                color: white;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        self.term.setReadOnly(True)
        
        # Заменяем label_3 на term в layout
        if hasattr(self, 'label_3'):
            # Находим позицию label_3 в layout и заменяем его
            layout = self.term_wi.layout()
            if layout:
                # Находим индекс label_3 в layout
                for i in range(layout.count()):
                    widget = layout.itemAt(i).widget()
                    if widget == self.label_3:
                        layout.removeWidget(self.label_3)
                        self.label_3.deleteLater()
                        layout.addWidget(self.term)
                        break
        
        # Перенаправляем стандартный вывод
        sys.stdout = StreamToTerminal(self.term)
        sys.stderr = StreamToTerminal(self.term)
        
        print("Программа запущена")
        print("Готов к работе...")
        
        # There are define models
        print("Загрузка моделей...")
        self.det_model = YOLO('main_models/det.pt')
        self.seg_model = YOLO('main_models/seg.pt')
        
        # Загружаем модель для Grad-CAM
        self.egfr_gradcam_model = None
        self.target_layers = None
        
        # Setup buttons
        self.ext_btn.clicked.connect(self.exit)
        self.browse_btn.clicked.connect(self.browse)
        self.process_btn.clicked.connect(self.process)
        self.comboBox.activated.connect(self.update_image)
        self.grad_cam_checkBox.stateChanged.connect(self.update_image)
        
        # Словари для хранения данных
        self.XY_dict = {}
        self.mutation_dict = {}
        self.cropped_images_dict = {}  # Храним вырезанные области
        
        # Set combo box
        self.statusBar.showMessage("Ready")

    def browse(self):
        self.listWidget.clear()
        self.comboBox.clear()
        self.data_files = list()
        self.cropped_images_dict.clear()

        self.directory = str(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.line_getcwd.setText('...'+str(self.directory)[-63:])
        
        if self.directory:
            for file in os.listdir(self.directory):
                if os.path.isdir(file) == False:
                    filename, file_extension = os.path.splitext(file)
                    if file_extension == '.jpg':
                        if file != '.DS_Store':
                            self.data_files.append(file)
                            self.listWidget.addItem(
                                '...'+str(self.directory)[-24:]+'/'+file)

            self.comboBox.addItems(self.data_files)
        self.statusBar.showMessage("Ready")
        print(f"Выбрана директория: {self.directory}")
        print(f"Найдено {len(self.data_files)} изображений")

    def extract_contour_region(self, image, mask, padding=50):
        """Извлекает область вокруг контура с отступом"""
        # Находим контуры на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # Если контуров нет, возвращаем центр изображения
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            size = 256  # Уменьшенный размер
            start_x = max(center_x - size//2, 0)
            start_y = max(center_y - size//2, 0)
            end_x = min(start_x + size, width)
            end_y = min(start_y + size, height)
            return image[start_y:end_y, start_x:end_x], []
        
        # Берем самый большой контур
        contour = max(contours, key=cv2.contourArea)
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        
        # Добавляем отступ
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)
        
        # Вырезаем область
        cropped = image[y:y+h, x:x+w]
        
        # Смещаем контур для вырезанной области
        shifted_contour = contour - np.array([x, y])
        
        return cropped, [shifted_contour]

    def process(self):
        print("\n" + "="*50)
        print("Начало обработки изображений...")
        print("="*50)
        
        self.XY_dict = {}
        self.mutation_dict = {}
        self.cropped_images_dict.clear()

        self.path_det = self.directory + '/detected'
        if not os.path.exists(self.path_det):
            os.mkdir(self.path_det)

        self.path_seg = self.directory + '/segmentated'
        if not os.path.exists(self.path_seg):
            os.mkdir(self.path_seg)

        image_count = 0
        for address, dirs, names in os.walk(self.directory):
            for name in names:
                filename, file_extension = os.path.splitext(name)
                if file_extension == ".jpg":
                    if name != '.DS_Store':
                        image_count += 1
                        print(f"\nОбработка изображения {image_count}: {name}")
                        
                        # Детекция
                        print("  - Детекция...")
                        results = self.det_model(self.directory+'/'+name)
                        
                        for r in results:
                            im_array = r.plot()
                            im = Image.fromarray(im_array[..., ::-1])
                            im.save(self.path_det + '/' + filename +
                                    '_DET_result.png')

                        im = cv2.imread(self.directory+'/'+name)
                        results = self.seg_model(im)

                        if results[0].masks is not None:
                            mask_raw = np.zeros(
                                (512, 512, 1), dtype=np.float32)
                            boxes = []
                            for item in results[0]:
                                mask_raw = mask_raw + \
                                    item.masks[0].cpu().data.numpy(
                                    ).transpose(1, 2, 0)
                                boxes.append(item.boxes[0].cpu().data.numpy())

                            for box in boxes:
                                self.x_center = np.round(
                                    (box[0][0]+box[0][2])/2)
                                self.y_center = np.round(
                                    (box[0][1]+box[0][3])/2)
                                self.XY_dict[name] = [
                                    self.x_center, self.y_center]
                            mask_raw = cv2.normalize(
                                mask_raw, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            contours, _ = cv2.findContours(
                                mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
                            # Save as mask
                            cv2.imwrite(str(self.path_seg) + '/' +
                                        filename + '_mask.png', mask_raw)
                            # Save masked for showing
                            cv2.imwrite(str(self.path_seg) + '/' +
                                        filename + '_masked.png', im)
                            
                            print(f"    Найдена мутация в координатах: X={self.x_center}, Y={self.y_center}")

                        # ВАЖНО: Вырезаем область вокруг контура, а не прямоугольную область
                        frame = cv2.imread(self.directory+'/'+name)
                        
                        # Извлекаем область вокруг контура мутации
                        img_cropped, contour_cropped = self.extract_contour_region(frame, mask_raw, padding=50)
                        
                        # Создаем копию для отображения с контуром
                        img_with_contour = img_cropped.copy()
                        
                        # Рисуем контур на вырезанной области (если есть)
                        if contour_cropped:
                            cv2.drawContours(img_with_contour, contour_cropped, -1, (0, 255, 0), 2)
                        
                        # Сохраняем обе версии
                        self.cropped_images_dict[name] = {
                            'with_contour': cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB),  # С контуром
                            'original': cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)  # Без контура для Grad-CAM
                        }

                        Resolution = (256, 256)
                        preprocess = transforms.Compose(
                            [
                                transforms.Resize(Resolution),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                     0.229, 0.224, 0.225]),
                            ]
                        )

                        classes = {0: "mutant", 1: "wildtype"}

                        model = torch.load(
                            'mutation_models/model_EGFR.pt', map_location=torch.device('cpu'), weights_only=False)
                        model.eval()

                        device = 'cpu'

                        if isinstance(img_cropped, np.ndarray):
                            if img_cropped.dtype != np.uint8:
                                img_cropped = img_cropped.astype(np.uint8)
                            img_pil = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

                        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                        output = model(img_tensor).argmax(dim=1).to('cpu').numpy()
                        self.EGFR_result = classes[output[0]]

                        model = torch.load(
                            'mutation_models/model_KRAS.pt', map_location=torch.device('cpu'), weights_only=False)
                        model.eval()

                        device = 'cpu'

                        output = model(img_tensor).argmax(dim=1).to('cpu').numpy()
                        self.KRAS_result = classes[output[0]]
                        self.mutation_dict[name] = [
                            self.EGFR_result, self.KRAS_result]
                        
                        print(f"    Результаты: EGFR={self.EGFR_result}, KRAS={self.KRAS_result}")
                        
                        # Сохраняем вырезанную область
                        cropped_save_path = os.path.join(self.path_seg, f"{filename}_cropped.png")
                        cv2.imwrite(cropped_save_path, img_with_contour)
                        
                        print(f"  ✓ Изображение обработано")
        print("\n" + "="*50)
        print(f"Обработка завершена! Обработано изображений: {image_count}")
        print("="*50)
        self.statusBar.showMessage("Images saved")

    def get_gradcam_for_image(self, img_cropped):
        """Генерация Grad-CAM для вырезанной области"""
        try:
            # Если модель еще не загружена, загружаем
            if self.egfr_gradcam_model is None:
                print("Загрузка модели для Grad-CAM...")
                self.egfr_gradcam_model = torch.load(
                    'mutation_models/model_EGFR.pt', 
                    map_location=torch.device('cpu'), 
                    weights_only=False
                )
                self.egfr_gradcam_model.eval()
                
                # Определяем целевые слои (зависит от архитектуры модели)
                # Если это ResNet
                if hasattr(self.egfr_gradcam_model, 'layer4'):
                    self.target_layers = [self.egfr_gradcam_model.layer4[-1]]
                # Если это другая архитектура, попробуем найти подходящие слои
                else:
                    # Ищем последний сверточный слой
                    target_layers = []
                    for name, module in self.egfr_gradcam_model.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            target_layers.append(module)
                    if target_layers:
                        self.target_layers = [target_layers[-1]]
                    else:
                        print("Не удалось найти подходящие слои для Grad-CAM")
                        return None
            
            # Препроцессинг
            IMAGE_SIZE = (256, 256)
            preprocess_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Конвертируем numpy array в PIL Image
            if isinstance(img_cropped, np.ndarray):
                img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                rgb_img = Image.fromarray(img_cropped_rgb).convert('RGB')
            else:
                rgb_img = img_cropped.convert('RGB')
            
            # Подготавливаем входные данные
            input_tensor = preprocess_transform(rgb_img).unsqueeze(0)
            
            # Создаем Grad-CAM
            cam = GradCAM(model=self.egfr_gradcam_model, 
                         target_layers=self.target_layers)
            
            # Генерируем тепловую карту
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            
            # Преобразуем исходное изображение для наложения
            rgb_img_np = np.array(rgb_img.resize(IMAGE_SIZE)) / 255.0
            
            # Создаем визуализацию
            visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
            
            return visualization
            
        except Exception as e:
            print(f"Ошибка при создании Grad-CAM: {str(e)}")
            return None

    def update_image(self):
        current_file = str(self.comboBox.currentText())
        if not current_file:
            return
            
        self.filename_changed_file, self.file_extension_changed_file = os.path.splitext(current_file)
        
        # Отображаем исходное изображение
        raw_pixmap = QPixmap(self.directory+'/'+self.filename_changed_file+'.jpg')
        scaled_raw_pixmap = raw_pixmap.scaled(self.raw_image.size(), 
                                            QtCore.Qt.KeepAspectRatio,
                                            QtCore.Qt.SmoothTransformation)
        self.raw_image.setPixmap(scaled_raw_pixmap)
        
        try:
            # Отображаем детектированное изображение
            det_pixmap = QPixmap(self.path_det + '/' + self.filename_changed_file +
                                '_DET_result.png')
            scaled_det_pixmap = det_pixmap.scaled(self.mask_l.size(), 
                                                QtCore.Qt.KeepAspectRatio,
                                                QtCore.Qt.SmoothTransformation)
            self.mask_l.setPixmap(scaled_det_pixmap)
            
            # Проверяем, есть ли вырезанная область
            if current_file in self.cropped_images_dict:
                cropped_data = self.cropped_images_dict[current_file]
                
                # Проверяем состояние чекбокса Grad-CAM
                if self.grad_cam_checkBox.isChecked():
                    print(f"Генерация Grad-CAM для {current_file}...")
                    
                    # Получаем оригинальное вырезанное изображение (без контура)
                    original_cropped_rgb = cropped_data['original']
                    # Конвертируем RGB в BGR для Grad-CAM
                    original_cropped_bgr = cv2.cvtColor(original_cropped_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Генерируем Grad-CAM
                    gradcam_result = self.get_gradcam_for_image(original_cropped_bgr)
                    
                    if gradcam_result is not None:
                        # Конвертируем в QPixmap
                        height, width, channel = gradcam_result.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(gradcam_result.data, width, height, 
                                      bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        
                        # Масштабируем с ограничением увеличения
                        scaled_pixmap = self.scale_pixmap_to_fit(pixmap, self.segm_l.size())
                        self.segm_l.setPixmap(scaled_pixmap)
                        print("✓ Grad-CAM сгенерирован")
                    else:
                        # Если Grad-CAM не удалось, показываем вырезанную область с контуром
                        self.show_cropped_image(cropped_data['with_contour'])
                else:
                    # Если чекбокс выключен, показываем вырезанную область С КОНТУРОМ
                    self.show_cropped_image(cropped_data['with_contour'])
            else:
                # Пытаемся показать сегментированное изображение
                segm_path = str(self.path_seg) + '/' + self.filename_changed_file + '_masked.png'
                if os.path.exists(segm_path):
                    segm_pixmap = QPixmap(segm_path)
                    scaled_segm_pixmap = self.scale_pixmap_to_fit(segm_pixmap, self.segm_l.size())
                    self.segm_l.setPixmap(scaled_segm_pixmap)
            
            # Обновляем координаты
            if current_file in self.XY_dict:
                self.x_loc_label.setText(
                    str(self.XY_dict[current_file][0]))
                self.y_loc_label.setText(
                    str(self.XY_dict[current_file][1]))

            # Обновляем результаты мутаций
            if current_file in self.mutation_dict:
                self.kras_label.setText(
                    str(self.mutation_dict[current_file][1]))
                self.egfr_label.setText(
                    str(self.mutation_dict[current_file][0]))
                
        except AttributeError as e:
            error_msg = f"medSeg ~ % The mask, segmented images and parameters\nmedSeg ~ % will be displayed after processing\nОшибка: {str(e)}"
            self.term.append(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"Ошибка при обновлении изображения: {str(e)}"
            self.term.append(error_msg)
            print(error_msg)

    def scale_pixmap_to_fit(self, pixmap, target_size):
        """Масштабирование пиксмапа для заполнения целевого размера с сохранением пропорций"""
        original_size = pixmap.size()
        target_width = target_size.width()
        target_height = target_size.height()
        
        # Вычисляем коэффициенты масштабирования
        scale_w = target_width / original_size.width()
        scale_h = target_height / original_size.height()
        
        # Используем минимальный коэффициент, чтобы изображение поместилось целиком
        scale = min(scale_w, scale_h)
        
        # Вычисляем новые размеры
        new_width = int(original_size.width() * scale)
        new_height = int(original_size.height() * scale)
        
        return pixmap.scaled(new_width, new_height, 
                           QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)

    def show_cropped_image(self, img_np):
        """Отображение вырезанной области с ограниченным увеличением"""
        if isinstance(img_np, np.ndarray):
            # Конвертируем numpy array в QPixmap
            height, width, channel = img_np.shape
            bytes_per_line = 3 * width
            
            # Проверяем формат
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
            
            # Создаем QImage
            q_img = QImage(img_np.data, width, height, 
                          bytes_per_line, QImage.Format_RGB888)
            
            # Создаем QPixmap
            pixmap = QPixmap.fromImage(q_img)
            
            # Масштабируем для заполнения
            scaled_pixmap = self.scale_pixmap_to_fit(pixmap, self.segm_l.size())
            self.segm_l.setPixmap(scaled_pixmap)

    def exit(self):
        print("Завершение работы программы...")
        self.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()