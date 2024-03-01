# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import desing
import numpy as np
from PIL import Image
from mymodel import net
from PyQt5 import QtWidgets
from ultralytics import YOLO
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageFile
from torchvision import transforms



class ExampleApp(QtWidgets.QMainWindow, desing.Ui_MainWindow):

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
        # Set combo box
        self.statusBar.showMessage("Ready")

    def browse(self):
        self.listWidget.clear()
        self.comboBox.clear()

        self.data_files = list()

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

    def process(self):
        self.XY_dict = {}
        self.mutation_dict = {}

        self.path_det = self.directory + '/detected'
        if not os.path.exists(self.path_det):
            os.mkdir(self.path_det)

        self.path_seg = self.directory + '/segmentated'
        if not os.path.exists(self.path_seg):
            os.mkdir(self.path_seg)

        for address, dirs, names in os.walk(self.directory):
            for name in names:
                filename, file_extension = os.path.splitext(name)
                if file_extension == ".jpg":
                    if name != '.DS_Store':

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
                        'mutation_models/model_EGFR.pt', map_location=torch.device('cpu'))
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
                        'mutation_models/model_KRAS.pt', map_location=torch.device('cpu'))
                    model.eval()

                    device = 'cpu'

                    output = model(img).argmax(dim=1).to('cpu').numpy()
                    self.KRAS_result = classes[output[0]]
                    self.mutation_dict[name] = [
                        self.EGFR_result, self.KRAS_result]
        self.statusBar.showMessage("Images saved")

    def update_image(self):
        self.term.clear()

        self.filename_changed_file, self.file_extension_changed_file = os.path.splitext(
            str(self.comboBox.currentText()))
        self.raw_image.setPixmap(
            QPixmap(self.directory+'/'+self.filename_changed_file+'.jpg'))
        try:
            self.mask_l.setPixmap(QPixmap(self.path_det + '/' + self.filename_changed_file +
                                          '_DET_result.png'))
            self.segm_l.setPixmap(QPixmap(str(self.path_seg) + '/' +
                                          self.filename_changed_file + '_masked.png'))
            self.x_loc_label.setText(
                str(self.XY_dict[self.comboBox.currentText()][0]))
            self.y_loc_label.setText(
                str(self.XY_dict[self.comboBox.currentText()][1]))

            self.kras_label.setText(
                str(self.mutation_dict[self.comboBox.currentText()][1]))
            self.egfr_label.setText(
                str(self.mutation_dict[self.comboBox.currentText()][0]))
        except AttributeError:
            self.term.setText(
                "medSeg ~ % The mask,  segmented images and parameters\nmedSeg ~ % will be displayed after processing")
            pass

    def exit(self):
        exit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
