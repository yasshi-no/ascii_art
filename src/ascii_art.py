#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ascii artで動画を作成するプログラム"""

import copy
import os
import sys
import time
import winsound

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


_FILE_DIR = os.path.dirname(__file__)                       # 実行ファイルの絶対パス
_TEMP_DIR = os.path.join(_FILE_DIR, "_temp")                # 一時ファイルのディレクトリ
# ConversionTableのための一時ファイルのディレクトリ
_TEMP_CHRIMG_DIR = os.path.join(_TEMP_DIR, "_chrimg")
# AsciiMovのための一時ファイルのディレクトリ
_TEMP_MOV_DIR = os.path.join(_TEMP_DIR, "_mov")
_TEMP_MOV_PATH = os.path.join(
    _TEMP_MOV_DIR, "_mov.txt")    # AsciiMovのための一時ファイル


def measure(func):
    """実行時間計測用のデコレーター"""
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print(t2-t1)
        return res
    return wrapper


class Progress:
    """プロセスの進行を出力するためのクラス"""

    def __init__(self, message, qty, cnt=0):
        self.message = message
        self._qty = qty  # 全体の作業量
        self._cnt = cnt  # 完了した作業量
        self._bar_qty = 20  # 作業量表示のメモリの数

    def print_mes(self):
        """messageを表示"""
        sys.stdout.write(self.message + "\n")

    def print_bar(self, cnt=1, end=False):
        """進捗を表示"""
        self._cnt += cnt
        ratio = self._cnt / self._qty  # 完了した仕事の割合
        message = "\r"

        # 目盛り部分の作成
        br = "|"*int(ratio / (1/self._bar_qty))
        br += "."*(self._bar_qty - len(br))
        message += br

        # 数字表示の作成
        message += "  {}/{}".format(self._cnt, self._qty)
        sys.stdout.write(message)

        if self._cnt == self._qty or end:
            sys.stdout.write("\n")


class ConvertionTable:
    """各明るさを変換方法に関するクラス"""

    def __init__(self):
        self._table = None  # 明るさから文字に変換する配列
        os.makedirs(_TEMP_CHRIMG_DIR, exist_ok=True)

    @property
    def table(self):
        """安全な_tableへのアクセス, coppy推奨"""
        res = copy.copy(self._table)
        return res

    def calc_img_from_chr(self, char, font, shape=(128, 128), pos=(0, 0), fill=255):
        """文字を画像に変換する"""
        img = np.zeros(shape)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, char, font=font, fill=fill)
        img = np.array(img_pil)
        return img

    def _store_imgs_from_chars(self, chars, font):
        """charsを画像に変換して保存する"""
        for i, char in enumerate(chars):
            path = os.path.join(_TEMP_CHRIMG_DIR, str(i)+".png")
            img = self.calc_img_from_chr(char, font)
            cv2.imwrite(path, img)

    def count_white(self, img):
        """白いピクセルを数える"""
        res = np.sum(img) // 255
        return res

    def create_table_from_font(self, chars, font_path):
        """fontから_tableを作成する"""
        num = 256  # 明るさの数
        font = ImageFont.truetype(font_path, 96)
        white_qties = [0]*len(chars)  # 各文字の白いピクセルの数
        nearests = [(float("inf"), None)]*num
        self._store_imgs_from_chars(chars, font)

        # 各文字の白いピクセルを数える
        for i in range(len(chars)):
            img = cv2.imread(os.path.join(_TEMP_CHRIMG_DIR,
                                          str(i)+".png"), cv2.COLOR_BGR2GRAY)
            white_qties[i] = self.count_white(img)

        # 白いピクセルの数が最大のものが(num-1)になるように正規化
        mx = max(white_qties)
        for i in range(len(chars)):
            white_qties[i] = int(white_qties[i]*(num-1)/mx)

        # 各明るさが正規化された白いピクセルの数で最も近い文字に変換されるようにする
        for i, char in enumerate(chars):
            for j in range(num):
                dev = (white_qties[i] - j)**2
                if dev < nearests[j][0]:
                    nearests[j] = (dev, char)

        self._table = [nearests[i][1] for i in range(num)]


class AsciiArt:
    """ascii artに関する親クラス"""

    def __init__(self):
        self._letter_height = None  # ascii artの高さ(文字数)
        self._letter_width = None  # ascii artの幅(文字数)

    @property
    def shape(self):
        """安全な_shapeへのアクセス"""
        return (self._height, self._width)

    @staticmethod
    def calc_ratio(shape):
        """letter_heightに対するletter_widthの比を求める"""
        height, width = shape
        ratio = width * 2 / height  # 文字の縦は横の2倍なので
        return ratio

    @staticmethod
    def calc_letter_shape(shape, letter_height):
        """shapeとletter_heightからletter_shapeを求める"""
        letter_width = int(letter_height * AsciiArt.calc_ratio(shape))
        return (letter_height, letter_width)

    def gen_ascii(self, img, letter_shape, cvt, white_back=False):
        """ascii artを作成"""
        self._letter_height, self._letter_width = letter_shape
        lst = [[""]*self._letter_width for i in range(self._letter_height)]
        img = cv2.resize(img, (self._letter_width, self._letter_height))
        table = cvt.table

        # 各画素を変換
        for y in range(self._letter_height):
            for x in range(self._letter_width):
                mean = img[y][x]
                if white_back:
                    mean = 255 - mean
                lst[y][x] = table[mean]

        res = ""
        for y in range(self._letter_height):
            res = res + "".join(lst[y]) + "\n"

        return res


class AsciiMov:
    """動画をascii artにするためのクラス"""

    def __init__(self, movie_path, letter_height):
        self._movie_path = movie_path  # 動画へのパス

        # 動画のデータ読み込み
        mov = cv2.VideoCapture(movie_path)
        self._height, self._width = int(mov.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            mov.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._fps = mov.get(cv2.CAP_PROP_FPS)
        self._frame_qty = int(mov.get(cv2.CAP_PROP_FRAME_COUNT))
        mov.release()

        self._letter_height = letter_height
        _, self._letter_width = AsciiArt.calc_letter_shape(
            self.shape, self._letter_height)
        self._init_temp_mov()

    @property
    def shape(self):
        """安全な_shapeへのアクセス"""
        return (self._height, self._width)

    @property
    def letter_shape(self):
        """安全な_letter_shapeへのアクセス"""
        return (self._letter_height, self._letter_width)

    def _init_temp_mov(self):
        """一時ディレクトリの作成"""
        os.makedirs(_TEMP_MOV_DIR, exist_ok=True)

    def calc_ascii_mov(self, cvt):
        """ascii artを作成し, _TEMP_MOV_PATHに保存する"""
        mov = cv2.VideoCapture(self._movie_path)
        pros = Progress("calc_ascii_mov", self._frame_qty)  # 作業進行度を表すオブジェクト
        pros.print_mes()
        with open(_TEMP_MOV_PATH, mode="w") as f:
            for i in range(self._frame_qty):
                _, frame = mov.read()  # フレームの読み込み
                # フレームを文字に変換
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ascii_frame = AsciiArt()
                ascii_frame = ascii_frame.gen_ascii(
                    img, self.letter_shape, cvt)

                f.write(ascii_frame)

                # 一定周期で進行度を出力
                if i % 10 == 0:
                    pros.print_bar(cnt=10)
                elif i == self._frame_qty-1:
                    pros.print_bar(cnt=self._frame_qty % 10)

    def print_mov_with_sound(self, sound_path):
        """音声を再生しながらprint_movを実行"""
        winsound.PlaySound(
            sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.print_mov()

    def print_mov(self):
        """計算したascii artを再生する"""
        os.system("mode {}, {}".format(
            self._letter_width, self._letter_height+2))

        # fps計算のための変数
        t1 = time.time()
        t2 = 0
        dt = 1/self._fps

        with open(_TEMP_MOV_PATH, mode="r") as f:
            for i in range(self._frame_qty):
                # 1フレーム文の出力の作成
                message = "\n"
                for j in range(self._letter_height):
                    message += f.readline()
                message += "{}/{}, {:.1f} fps\n".format(
                    i+1, self._frame_qty, (i)/dt)

                sys.stdout.write(message)

                # 出力に要した時間を差し引いて, 次のフレームまで休止
                t2 = time.time()
                dt = t2 - t1
                pose = max(((i+1)/self._fps)-dt, 0)
                time.sleep(pose)
                t2 = time.time()
                dt = t2 - t1


class AsciiCam:
    """動画をリアルタイムでアスキーアートに変換するためのクラス"""

    def __init__(self, cap, letter_height):
        super().__init__()
        self._cap = cap

        # カメラの情報読み取り
        self._height, self._width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._letter_height = letter_height
        self._letter_height, self._letter_width = AsciiArt.calc_letter_shape(
            self.shape, self._letter_height)

    @property
    def shape(self):
        """安全な_shapeへのアクセス"""
        return (self._height, self._width)

    @property
    def letter_shape(self):
        """安全な_letter_shapeへのアクセス"""
        return (self._letter_height, self._letter_width)

    def print_mov(self, cvt, fps=30):
        """カメラからの入力をリアルタイムで変換し出力する"""
        # コマンドプロンプトのサイズを変更
        os.system("mode {}, {}".format(
            self._letter_width, self._letter_height+2))

        # fps計算用の変数
        t1 = time.time()
        t2 = time.time()
        dt = 1/fps
        pose = 0

        while True:
            _, frame = self._cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ascii_frame = AsciiArt()

            # 1フレームの出力を作成
            message = "\n"
            message += ascii_frame.gen_ascii(img, (self._letter_height,
                                                   self._letter_width), cvt)
            message += "{:.1f} fps\n".format(1/(dt+pose))

            # 出力に要した時間を差し引いて, 次のフレームまで待機
            sys.stdout.write(message)
            t2 = time.time()
            dt = t2 - t1
            pose = max((1/fps)-dt, 0)
            time.sleep(pose)
            t1 = t2


def main():
    """main関数"""
    font_path = "C:/Windows/Fonts/msgothic.ttc"
    # movie_path = "bad apple.mp4"
    # sound_path = "bad apple.wav"
    movie_path = "lagtrain.mp4"
    sound_path = "lagtrain.wav"

    cvt = ConvertionTable()
    chars = [chr(i) for i in range(32, 127)]
    cvt.create_table_from_font(chars, font_path)
    cvt._table[255] = "@"

    asc_mov = AsciiMov(movie_path, 45)
    asc_mov.calc_ascii_mov(cvt)
    asc_mov.print_mov_with_sound(sound_path)

    # cap = cv2.VideoCapture(0)
    # ascii_cam = AsciiCam(cap, 50)
    # ascii_cam.print_mov(cvt)


if __name__ == "__main__":
    main()
