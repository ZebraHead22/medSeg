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
from PyQt5 import QtWidgets
from ultralytics import YOLO
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageFile
from torchvision import transforms
# from radiomics import featureextractor
from PyQt5 import QtGui  # Добавьте этот импорт в начало файла
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



class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = os.getcwd()
        self.data = None
        # There are define models
        self.det_model = YOLO('main_models/det.pt')
        self.seg_model = YOLO('main_models/seg.pt')
        # Setup buttons
        self.ext_btn.clicked.connect(self.exit)
        self.browse_btn.clicked.connect(self.browse)
        self.process_btn.clicked.connect(self.process)
        self.comboBox.activated.connect(self.update_image)
        
        # Подключаем сигнал чекбокса Grad-CAM
        self.grad_cam_checkBox.stateChanged.connect(self.update_image)
        
        # Устанавливаем стиль для чекбокса
        self.grad_cam_checkBox.setStyleSheet("""
            QCheckBox {
                color: white;
                background-color: rgb(66, 71, 105);
                border-radius: 5px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
        """)
        
        # Set combo box
        self.statusBar.showMessage("Ready")
        self.bbox_dict = {}  # Словарь для хранения bounding box опухолей
        # Инициализируем терминальное окно
        self.term = self.label_3  # Используем label_3 как терминальное окно

    def browse(self):
        self.listWidget.clear()
        self.comboBox.clear()

        self.data_files = list()

        self.directory = str(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.line_getcwd.setText('...'+str(self.directory)[-63:])
        
        # Вывод в терминальное окно
        self.term.setText("Selected directory: " + self.directory)
        
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
            # Вывод в терминальное окно
            self.term.setText(f"Found {len(self.data_files)} images\n" + self.term.text())
            
        self.statusBar.showMessage("Ready")

    def process(self):
        # Вывод в терминальное окно
        self.term.setText("Starting processing...\n" + self.term.text())
        
        self.XY_dict = {}
        self.mutation_dict = {}
        self.bbox_dict = {}  # Очищаем словарь bounding box'ов

        self.path_det = self.directory + '/detected'
        if not os.path.exists(self.path_det):
            os.mkdir(self.path_det)
            self.term.setText("Created 'detected' directory\n" + self.term.text())

        self.path_seg = self.directory + '/segmentated'
        if not os.path.exists(self.path_seg):
            os.mkdir(self.path_seg)
            self.term.setText("Created 'segmentated' directory\n" + self.term.text())

        image_count = 0
        processed_count = 0
        
        for address, dirs, names in os.walk(self.directory):
            for name in names:
                filename, file_extension = os.path.splitext(name)
                if file_extension == ".jpg":
                    image_count += 1
                    if name != '.DS_Store':
                        # Вывод в терминальное окно
                        self.term.setText(f"Processing image {image_count}: {name}\n" + self.term.text())
                        QtWidgets.QApplication.processEvents()  # Обновляем интерфейс

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
                                
                                # Сохраняем bounding box для каждой опухоли
                                x_min = int(box[0][0])
                                y_min = int(box[0][1])
                                x_max = int(box[0][2])
                                y_max = int(box[0][3])
                                
                                # УВЕЛИЧИВАЕМ padding для большей области
                                padding = 20  # Увеличили с 20 до 50
                                height, width = im.shape[:2]
                                x_min = max(0, x_min - padding)
                                y_min = max(0, y_min - padding)
                                x_max = min(width, x_max + padding)
                                y_max = min(height, y_max + padding)
                                
                                # Сохраняем в словарь
                                if name not in self.bbox_dict:
                                    self.bbox_dict[name] = []
                                self.bbox_dict[name].append({
                                    'x_min': x_min,
                                    'y_min': y_min,
                                    'x_max': x_max,
                                    'y_max': y_max
                                })
                                
                            mask_raw = cv2.normalize(
                                mask_raw, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            contours, _ = cv2.findContours(
                                mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # СДЕЛАЛИ ОБВОДКУ ТОНЬШЕ (1 вместо 2)
                            cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
                            
                            # Сохраняем оригинальное изображение с контуром
                            cv2.imwrite(str(self.path_seg) + '/' +
                                        filename + '_masked_full.png', im)
                            
                            # Создаем и сохраняем вырезанные области для каждой опухоли
                            if name in self.bbox_dict:
                                for i, bbox in enumerate(self.bbox_dict[name]):
                                    # Вырезаем область опухоли
                                    tumor_region = im[bbox['y_min']:bbox['y_max'], 
                                                     bbox['x_min']:bbox['x_max']]
                                    
                                    # Сохраняем вырезанную область
                                    if len(self.bbox_dict[name]) == 1:
                                        # Если опухоль одна, сохраняем с обычным именем
                                        cv2.imwrite(str(self.path_seg) + '/' +
                                                    filename + '_masked.png', tumor_region)
                                    else:
                                        # Если опухолей несколько, сохраняем с индексом
                                        cv2.imwrite(str(self.path_seg) + '/' +
                                                    filename + f'_masked_{i}.png', tumor_region)
                            
                            # Save as mask
                            cv2.imwrite(str(self.path_seg) + '/' +
                                        filename + '_mask.png', mask_raw)

                        half_width, half_height = 64, 64
                        # Center is determined in get_mask, here x of the center and y of the center should be
                        start_x = int(max(self.x_center - half_width, 0))
                        start_y = int(max(self.y_center - half_height, 0))
                        end_x = int(start_x + 2*half_width)
                        end_y = int(start_y + 2*half_height)
                        frame = cv2.imread(self.directory+'/'+name)

                        # This is the main image CT
                        img = frame[start_y:end_y, start_x:end_x, :]

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

                        if isinstance(img, np.ndarray):
                            if img.dtype != np.uint8:
                                img = img.astype(np.uint8)
                            img = Image.fromarray(img)

                        img = preprocess(img).unsqueeze(0).to(device)
                        output = model(img).argmax(dim=1).to('cpu').numpy()
                        self.EGFR_result = classes[output[0]]

                        model = torch.load(
                            'mutation_models/model_KRAS.pt', map_location=torch.device('cpu'), weights_only=False)
                        model.eval()

                        device = 'cpu'

                        output = model(img).argmax(dim=1).to('cpu').numpy()
                        self.KRAS_result = classes[output[0]]
                        self.mutation_dict[name] = [
                            self.EGFR_result, self.KRAS_result]
                        
                        processed_count += 1
                        # Вывод в терминальное окно
                        self.term.setText(f"Processed {processed_count}/{image_count} images\n" + self.term.text())
                        QtWidgets.QApplication.processEvents()  # Обновляем интерфейс

        # Вывод в терминальное окно
        self.term.setText(f"✅ Processing complete!\nProcessed {processed_count} out of {image_count} images\n" + self.term.text())
        self.statusBar.showMessage(f"Images saved ({processed_count} processed)")

    def create_grad_cam(self, image_path, model_path='mutation_models/model_EGFR.pt'):
        """
        Создает тепловую карту Grad-CAM для изображения
        """
        try:
            # Используем вырезанную область опухоли, а не все изображение
            # Получаем путь к вырезанной области опухоли
            filename = os.path.splitext(os.path.basename(image_path))[0]
            tumor_image_path = str(self.path_seg) + '/' + filename + '_masked.png'
            
            if not os.path.exists(tumor_image_path):
                # Ищем другие варианты
                for i in range(10):
                    alt_path = str(self.path_seg) + '/' + filename + f'_masked_{i}.png'
                    if os.path.exists(alt_path):
                        tumor_image_path = alt_path
                        break
            
            if not os.path.exists(tumor_image_path):
                # Если нет вырезанной области, используем полное изображение
                self.term.setText("Using full image for Grad-CAM (no tumor region found)\n" + self.term.text())
                tumor_image_path = str(self.path_seg) + '/' + filename + '_masked_full.png'
                if not os.path.exists(tumor_image_path):
                    # Если нет и полного изображения, используем оригинал
                    tumor_image_path = image_path
            
            # Загружаем изображение
            rgb_img = Image.open(tumor_image_path).convert('RGB')
            
            # Определяем размер для модели (должен совпадать с размером, на котором обучалась модель)
            IMAGE_SIZE = (256, 256)
            
            # Определяем препроцессинг
            preprocess_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # Преобразуем изображение
            input_tensor = preprocess_transform(rgb_img).unsqueeze(0)
            
            # Загружаем модель
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            
            # Определяем target_layers (зависит от архитектуры модели)
            # Для стандартных ResNet моделей:
            if hasattr(model, 'layer4'):
                target_layers = [model.layer4[-1]]
            else:
                # Ищем последний сверточный слой
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        target_layers = [module]
                        break
            
            # Создаем Grad-CAM
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            
            # Преобразуем оригинальное изображение для визуализации
            rgb_img_np = np.array(rgb_img.resize(IMAGE_SIZE)) / 255.0
            
            # Создаем визуализацию
            visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
            
            # Конвертируем в формат для отображения в QLabel
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            return visualization_bgr
        
        except Exception as e:
            error_msg = f"Error creating Grad-CAM: {e}"
            print(error_msg)
            self.term.setText(error_msg + "\n" + self.term.text())
            return None
        

    def update_image(self):
        # self.term.clear()

        self.filename_changed_file, self.file_extension_changed_file = os.path.splitext(
            str(self.comboBox.currentText()))
        
        # Загружаем исходное изображение
        current_image_path = self.directory + '/' + self.filename_changed_file + '.jpg'
        self.raw_image.setPixmap(
            QPixmap(current_image_path))
        
        try:
            # Загружаем детекцию
            self.mask_l.setPixmap(QPixmap(self.path_det + '/' + self.filename_changed_file +
                                        '_DET_result.png'))
            
            # ПРОВЕРЯЕМ ЧЕКБОКС GRAD-CAM
            if self.grad_cam_checkBox.isChecked():
                # Если чекбокс включен, создаем и показываем Grad-CAM
                self.term.setText("Creating Grad-CAM visualization...\n" + self.term.text())
                grad_cam_image = self.create_grad_cam(current_image_path)
                
                if grad_cam_image is not None:
                    # Конвертируем numpy array в QImage
                    height, width, channel = grad_cam_image.shape
                    bytes_per_line = 3 * width
                    
                    # Исправляем формат - используем RGB вместо BGR
                    grad_cam_rgb = cv2.cvtColor(grad_cam_image, cv2.COLOR_BGR2RGB)
                    q_img = QImage(grad_cam_rgb.data, width, height, 
                                  bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.segm_l.setPixmap(pixmap)
                    self.term.setText("Showing Grad-CAM visualization\n" + self.term.text())
                else:
                    # Если Grad-CAM не удалось создать, показываем обычное изображение
                    self.show_segmentation_image()
            else:
                # Если чекбокс выключен, показываем обычное изображение
                self.show_segmentation_image()
            
            # Обновляем координаты и мутации
            self.update_coordinates_and_mutations()
            
        except AttributeError:
            self.term.setText(
                "medSeg ~ % The mask, segmented images and parameters\nmedSeg ~ % will be displayed after processing")
        except Exception as e:
            self.term.setText(f"Error updating image: {str(e)}\n" + self.term.text())

    def show_segmentation_image(self):
        """Показывает обычное изображение с сегментацией"""
        # Проверяем, есть ли вырезанная область опухоли
        tumor_image_path = str(self.path_seg) + '/' + self.filename_changed_file + '_masked.png'
        
        if os.path.exists(tumor_image_path):
            # Если есть вырезанная область, показываем ее
            self.segm_l.setPixmap(QPixmap(tumor_image_path))
        else:
            # Ищем другие варианты (если опухолей несколько)
            found = False
            for i in range(10):  # Проверяем до 10 возможных опухолей
                alt_path = str(self.path_seg) + '/' + self.filename_changed_file + f'_masked_{i}.png'
                if os.path.exists(alt_path):
                    self.segm_l.setPixmap(QPixmap(alt_path))
                    found = True
                    break
            
            if not found:
                # Если нет вырезанных областей, показываем полное изображение
                full_image_path = str(self.path_seg) + '/' + self.filename_changed_file + '_masked_full.png'
                if os.path.exists(full_image_path):
                    self.segm_l.setPixmap(QPixmap(full_image_path))
                else:
                    self.segm_l.clear()

    def update_coordinates_and_mutations(self):
        """Обновляет координаты и мутации"""
        current_image = self.comboBox.currentText()
        
        if current_image in self.XY_dict:
            self.x_loc_label.setText(str(self.XY_dict[current_image][0]))
            self.y_loc_label.setText(str(self.XY_dict[current_image][1]))
            self.term.setText(f"Tumor center: X={self.XY_dict[current_image][0]}, Y={self.XY_dict[current_image][1]}\n" + self.term.text())
        
        if current_image in self.mutation_dict:
            self.kras_label.setText(str(self.mutation_dict[current_image][1]))
            self.egfr_label.setText(str(self.mutation_dict[current_image][0]))
            self.term.setText(f"EGFR: {self.mutation_dict[current_image][0]}, KRAS: {self.mutation_dict[current_image][1]}\n" + self.term.text())

    def exit(self):
        # Вывод в терминальное окно
        self.term.setText("Exiting application...\n" + self.term.text())
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()