import sys
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import QSize, Qt, QSizeF, QRectF, QRect
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsTextItem, QGraphicsItemGroup
from PySide6.QtWidgets import QGraphicsRectItem


def draw_frame(header_text, header_color, body_text, body_color):
    frame = QGraphicsItemGroup()

    # Sample header
    header = QGraphicsRectItem(QRectF(-26, -14, 52, 28))
    header.setBrush(QBrush(header_color))
    header.setPos(0, -50)

    header_text = QGraphicsTextItem(header_text)
    header_text.setParentItem(header)
    font = header_text.font()
    font.setPointSize(10)
    header_text.setFont(font)
    header_text.setTextWidth(52)
    header_text.setHtml("<div align='center'>" + header_text.toHtml() + "</div>")
    header_text.setPos(-26, -14)

    # Sample Frame
    body = QGraphicsRectItem(QRectF(-26, -50, 52, 100))
    body.setBrush(QBrush(body_color))
    body.setPos(0, 14)

    body_text = QGraphicsTextItem(body_text)
    body_text.setParentItem(body)
    body_text.setDefaultTextColor(QColor('#2f4f4f'))
    font = body_text.font()
    font.setPointSize(12)
    font.setBold(True)
    body_text.setFont(font)
    body_text.setTextWidth(100)
    body_text.setHtml("<div align='center'>" + body_text.toHtml() + "</div>")
    body_text.setPos(-13, 50)
    body_text.setRotation(-90)

    frame.addToGroup(header)
    frame.addToGroup(body)
    return frame

def draw_block(width, height, text):
    frame = QGraphicsItemGroup()
    # Sample header
    header = QGraphicsRectItem(QRectF(-width/2, -height/4, width, height/2))
    header.setBrush(QBrush(QColor('#f5f5f5')))
    header.setPos(0, -height/4)

    header_text = QGraphicsTextItem('Block')
    header_text.setParentItem(header)
    header_text.setDefaultTextColor(QColor('#2f4f4f'))
    font = header_text.font()
    font.setPointSize(10)
    header_text.setFont(font)
    header_text.setTextWidth(width)
    header_text.setHtml("<div align='center'>" + header_text.toHtml() + "</div>")
    header_text.setPos(-width/2, -height/4)

    body = QGraphicsRectItem(QRectF(-width/2, -height/4, width, height/2))
    body.setBrush(QBrush(QColor('#C8BFE7')))
    body.setPos(0, height/4)

    body_text = QGraphicsTextItem(text)
    body_text.setParentItem(body)
    body_text.setDefaultTextColor(QColor('#2f4f4f'))
    font = body_text.font()
    font.setPointSize(10)
    body_text.setFont(font)
    body_text.setTextWidth(width)
    body_text.setHtml("<div align='center'>" + body_text.toHtml() + "</div>")
    body_text.setPos(-width/2, -height/4)

    frame.addToGroup(header)
    frame.addToGroup(body)
    return frame

def draw_trial(scene, samples):
    pos_x = 20
    for i, sample in enumerate(samples):
        sample_frame = draw_frame("Sample", QColor('#f5f5f5'), sample["sample_type"], QColor('#90ee90'))
        scene.addItem(sample_frame)
        sample_frame.setPos(pos_x, 300)
        pos_x += 52


def draw_trials(scene, samples):
    trial = []
    current_trial_id = -1
    unique_trials = set()
    for i, sample in enumerate(samples):
        if sample['trial_type'] != "":
            if(sample['trial_id'] != current_trial_id):
                current_trial_id = sample['trial_id']
                if trial and sample['trial_type'] not in unique_trials:
                    draw_trial(scene, trial)
                    unique_trials.add(sample['trial_type'])
                trial = []
            trial.append(sample)

    print(unique_trials)


def plot_samples(scene, samples):

    trial_list = []
    block_id = -1
    trial_frame_list = []

    frame_width = 52
    frame_height = 128
    pos_x = frame_width
    for i, sample in enumerate(samples):
        if sample['block_id'] != block_id:
            block_id = sample['block_id']
            if trial_frame_list: #draw block and dop trial
                trial_frame = draw_frame("Trial", QColor('#f5f5f5'), '...', QColor('#87cefa'))
                scene.addItem(trial_frame)
                trial_frame.setPos(pos_x, 100)
                pos_x += frame_width
                trial_frame_list.append(trial_frame)

                first = trial_frame_list[0]
                last = trial_frame_list[-1]
                width = last.x() + frame_width - first.x()
                x = first.x() + (last.x()-first.x())/2
                y = first.y()-frame_height/2 - 56/2
                trial_frame_list = []
                block_frame = draw_block(width, 56, trial_list[0][0])
                block_frame.setPos(x, y)
                scene.addItem(block_frame)
            trial_list = []

        if sample['trial_type'] == "":
            sample_frame = draw_frame("Sample", QColor('#f5f5f5'), sample["sample_type"], QColor('#90ee90'))
            scene.addItem(sample_frame)
            sample_frame.setPos(pos_x, 100)
            pos_x += frame_width
        elif (sample['block_type'], sample['trial_type']) not in trial_list:
            trial_frame = draw_frame("Trial", QColor('#f5f5f5'), sample["trial_type"], QColor('#87cefa'))
            scene.addItem(trial_frame)
            trial_frame.setPos(pos_x, 100)
            pos_x += frame_width
            trial_list.append((sample['block_type'], sample['trial_type']))
            trial_frame_list.append(trial_frame)



    scene.setSceneRect(QRectF(0,0, pos_x, 200))
    #scene.addRect(scene.sceneRect())


def plot_design(samples):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    scene = QGraphicsScene(0, 0, 400, 400)
    view = QGraphicsView(scene=scene, parent=None)

    view.show()
    plot_samples(scene, samples)
    #draw_trials(scene, samples)
    view.setMinimumSize(QSize(900, 300))
    view.raise_()
    #app.exec_()
    return view

