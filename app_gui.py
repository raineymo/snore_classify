import librosa
import pygame
import os
from PySide2.QtCore import QFile
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow, QLabel, QVBoxLayout, QWidget, QStackedWidget
from PySide2.QtUiTools import QUiLoader
from matplotlib import pyplot as plt
from qt_material import apply_stylesheet
from convert_to_wav_forui import convert_to_wav, get_file_extension, Denoise
from test_5 import test_acc
import pyaudio
import threading
import wave

global filepath,rec

# 录音
class Recorder:
    def __init__(self, chunk=1024, channels=1, rate=64000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    # 定义开始录音
    def start(self):
        threading._start_new_thread(self.__recording, ())

    # 定义录音
    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while self._running:
            data = stream.read(self.CHUNK)
            self._frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    # 定义停止
    def stop(self):
        self._running = False

    # 定义保存
    def save(self, filename):
        p = pyaudio.PyAudio()
        if os.path.exists(filename):
            os.remove(filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        qfile_stats = QFile("ui/stats.ui")
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.filepath = ""
        # page1函数绑定
        self.ui.uploadbutton.clicked.connect(self.getuploadfilepath)
        self.ui.playbutton.clicked.connect(self.play)
        self.ui.pausebutton.clicked.connect(self.pause)
        self.ui.upvolume.clicked.connect(self.vol_up)
        self.ui.downvolume.clicked.connect(self.vol_down)
        self.ui.showresult.clicked.connect(self.show_result1)
        # page2函数绑定
        self.ui.playbutton_2.clicked.connect(self.play)
        self.ui.pausebutton_2.clicked.connect(self.pause)
        self.ui.showresult_2.clicked.connect(self.show_result2)
        self.ui.recordbutton.clicked.connect(self.start_audio)
        self.ui.stopbutton.clicked.connect(self.stop_audio)
        # 页面转换
        self.stackedWidget = self.ui.findChild(QStackedWidget, "stackedWidget")
        self.ui.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.ui.pushButton_2.clicked.connect(self.on_pushButton_2_clicked)
        self.ui.pushButton_3.clicked.connect(self.on_pushButton_3_clicked)
        self.pause_state = 0
        pygame.init()
        pygame.mixer.music.set_volume(0.5)
        self.get_vol = pygame.mixer.music.get_volume()

    # 槽函数：点击Go to Page 1按钮
    def on_pushButton_clicked(self):
        self.stackedWidget.setCurrentIndex(0)

    # 槽函数：点击Go to Page 2按钮
    def on_pushButton_2_clicked(self):
        self.stackedWidget.setCurrentIndex(1)

    # 槽函数：点击Go to Page 3按钮
    def on_pushButton_3_clicked(self):
        self.stackedWidget.setCurrentIndex(2)

    # 上传文件
    def getuploadfilepath(self):  # 获得上传文件的路径
        global filepath
        filepath1, filetype = QFileDialog.getOpenFileName(QMainWindow(), "选择文件", "",
                                                          "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.flv)")  # 选择目录，返回选中的路径
        if filepath1:  # 确保文件路径不为空
            filepath = filepath1  # 设置文件的路径
            print(filepath1)
            if get_file_extension(filepath) == '.wav':
                filepath = Denoise(filepath)
            else:
                wav_file = convert_to_wav(filepath)  # 转成wav
                filepath = wav_file
                filepath = Denoise(filepath)
            self.ui.audiostate.setText("音频已上传！")
        else:
            print("未选择文件或取消选择")
        print("实际文件为：" + filepath)

    # 显示鼾声识别结果
    def show_result1(self):
        global filepath
        result = test_acc(filepath)  # 通过模型
        # # 显示音频波形图
        widget=self.show_waveform(filepath)
        self.ui.scroll_area.setWidget(widget)
        self.ui.scroll_area.setWidgetResizable(True)  # 使 QWidget 可调整大小以适应 QScrollArea
        if result == 't':
            self.ui.label2.setText("是")
        else:
            self.ui.label2.setText("否")

    def show_result2(self):
        global filepath
        result = test_acc(filepath)  # 通过模型
        # 显示音频波形图
        print("显示波形图")
        self.show_waveform(filepath)
        widget = self.show_waveform(filepath)
        self.ui.scroll_area_2.setWidget(widget)
        self.ui.scroll_area_2.setWidgetResizable(True)  # 使 QWidget 可调整大小以适应 QScrollArea
        if result == 't':
            self.ui.label2_2.setText("是")
        else:
            self.ui.label2_2.setText("否")

    # 播放
    def play(self):
        if filepath and os.path.isfile(filepath):  # 确保文件路径不为空且文件存在
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            if self.pause_state == 0:
                pygame.mixer.music.unpause()
                self.ui.audiostate.setText("音频正在播放")
                self.ui.audiostate_2.setText("音频正在播放")
                self.pause_state = 1
            self.show_vol()
        else:
            self.ui.audiostate.setText("无效的文件路径或文件不存在")

    # 显示音量
    def show_vol(self):
        vol = str(int(self.get_vol * 100)) + "%"
        self.ui.audiovolume.setText("音量：" + vol)

    # 暂停音频
    def pause(self):
        if self.pause_state == 1:
            pygame.mixer.music.pause()
            self.ui.audiostate.setText("音频播放已暂停")
            self.pause_state = 0

    # 提高音量
    def vol_up(self):
        self.get_vol += 0.1
        if self.get_vol >= 1:
            self.get_vol = 1
        pygame.mixer.music.set_volume(self.get_vol)
        self.show_vol()

    # 降低音量
    def vol_down(self):
        self.get_vol -= 0.1
        if self.get_vol <= 0:
            self.get_vol = 0
        pygame.mixer.music.set_volume(self.get_vol)
        self.show_vol()

    # 绘制波形图
    def show_waveform(self,filepath):
        if not filepath or not os.path.isfile(filepath):
            self.ui.audiostate.setText("无效的文件路径或文件不存在")
            return
            # 从音频文件中读取波形数据
        waveform, sr = librosa.load(filepath, sr=None, mono=True)
        # 绘制
        plt.figure(figsize=(12, 2))  # 调整图形大小以适应波形
        plt.plot(waveform)
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout()  # 确保标签不会重叠
        # 保存绘制的波形图到临时文件
        temp_file = "temp_waveform.png"
        plt.savefig(temp_file)
        # 关闭图形，释放资源
        plt.close()
        # 加载波形图的 QPixmap
        waveform_pixmap = QPixmap(temp_file)
        # 创建一个 QLabel 用于显示 QPixmap
        label = QLabel()
        label.setPixmap(waveform_pixmap)
        # 创建一个 QWidget，并设置其布局为 QVBoxLayout
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(label)
        # 将 QWidget 添加到 QScrollArea
        return widget

    # 开始录制
    def start_audio(self):
        global rec
        rec=Recorder()
        rec.start()
        self.ui.audiostate_2.setText("音频开始录制")

    # 暂停录制
    def stop_audio(self):
        global rec,filepath
        rec.stop()
        rec.save("./audio/test.wav")
        filepath="./audio/test.wav"
        filepath = Denoise(filepath)
        self.ui.audiostate_2.setText("音频已录制")

app = QApplication([])
stats = Stats()
stats.ui.show()
apply_stylesheet(app, theme='light_blue.xml', css_file='css/Qui.css')
app.exec_()
