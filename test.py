
import tensorflow as tf
from gensim.downloader import base_dir
from sklearn.preprocessing import MinMaxScaler
from sympy.benchmarks.bench_meijerint import timings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
import pandas as pd
import os
from django.conf import settings
from datetime import datetime
import json
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # 卷积层 + 批量归一化 + 激活函数 + 最大池化层
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 平坦层，将3D特征图展平为1D向量
    model.add(Flatten())

    # 全连接层 + Dropout
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # 根据类别数选择输出层
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))  # 二元分类
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))  # 多类分类
        loss = 'categorical_crossentropy'

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=loss,
                  metrics=['accuracy'])

    return model



def load_image_data(train_dir, img_size , num_classes, batch_size=32):
    # 使用 ImageDataGenerator 來自動加載和預處理圖片
    img_size=img_size[:2]

    # 创建一个图像数据生成器，用于对训练数据进行数据增强
    train_datagne = ImageDataGenerator(
        rescale=1. / 255,  # 将像素值缩放到0到1之间
        rotation_range=40,  # 随机旋转角度
        width_shift_range=0.2,  # 随机水平平移
        height_shift_range=0.2,  # 随机垂直平移
        shear_range=0.2,  # 随机错切变换
        zoom_range=0.2,  # 随机缩放
        horizontal_flip=True,  # 随机水平翻转
        fill_mode='nearest',  # 填充方式为最近邻填充
        validation_split = 0.2
    )

    # 创建一个图像数据生成器，仅对验证数据进行归一化处理
    test_datagne = ImageDataGenerator(rescale=1. / 255,validation_split=0.2 )
    # 訓練數據集
    train_generator = train_datagne.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical' if num_classes>2 else 'binary',  # 對應多類分類問題
        subset='training'  # 使用 80% 圖片作為訓練數據
    )

    # 驗證數據集
    validation_generator = test_datagne.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical' if num_classes>2 else 'binary',
        subset='validation'  # 使用 20% 圖片作為驗證數據
    )

    return train_generator, validation_generator

num_classes = 2  # 类别數
input_shape = (128, 128, 3)
base_dir1=r'C:\Users\MSI\Desktop\ai_analysis\media\uploads\test_dir'
# 使用範例
train_dir = os.path.join(base_dir1,os.listdir(base_dir1)[0])  # 使用者上傳的圖片數據夾路徑
train_generator, validation_generator = load_image_data(train_dir, img_size=input_shape ,num_classes=num_classes)



model = create_cnn_model(input_shape, num_classes)

history = model.fit(train_generator, epochs=10, batch_size=32, validation_data=validation_generator, verbose=2)

model.save(f'cnn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
# 从训练历史中提取训练准确率和验证准确率
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training_accuracy')  # 训练准确率为蓝色线
plt.plot(epochs, val_acc, 'r', label='Validation_accuracy')  # 验证准确率为红色线
plt.title('Training and Validation accuracy')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()

