from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sympy.benchmarks.bench_meijerint import timings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, \
    BatchNormalization, MultiHeadAttention, LayerNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop
from keras_tuner.tuners import RandomSearch
import pandas as pd
import os
from django.conf import settings
from datetime import datetime
import json
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import RandomizedSearchCV
from io import BytesIO
from .models import auth_model
import shutil
import uuid
import zipfile
import tempfile
import pickle


def make_lstm_plot(x_true, x_test, y_true, predictions, username, unique_id, is_train):
    # 繪製圖表
    # print('X:',x_test.shape,'y:',y_test_true.shape,'p:',predictions.shape)
    plt.plot(x_true, y_true, label='True Values', marker='o')
    plt.plot(x_test, predictions, label='Predictions', marker='x')
    plt.title('True Values vs Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    # 使用 MaxNLocator 自動控制 x 軸標籤數量
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=10))  # nbins 控制最多顯示的標籤數量

    plt.xticks(rotation=45, ha='right')  # 旋轉 x 軸標籤，避免擁擠
    plt.tight_layout()

    media_path = os.path.join(settings.MEDIA_ROOT, 'plots')

    # 3. 圖片檔案名稱 (避免重名使用 timestamp)
    filename = f"{username}_{unique_id}_keep_plot.png"
    file_path = os.path.join(media_path, filename)
    # 4. 儲存圖片
    plt.savefig(file_path)
    plt.close()

    # 5. 生成圖片的 URL
    image_url = f"{settings.MEDIA_URL}plots/{filename}"
    return image_url


# Create your views here.
def lstm_train(request, model_name=None):
    # 定義模型構建函數
    def build_random_lstm(hp):
        model = Sequential()

        # 隨機選擇LSTM層數和單元數
        for i in range(hp.Int('num_layers', 1, 3)):  # 隨機選擇1到3層LSTM
            model.add(LSTM(units=hp.Int('units_' + str(i),
                                        min_value=32,
                                        max_value=128,
                                        step=32),
                           return_sequences=(i < hp.Int('num_layers', 1, 3) - 1),
                           input_shape=(None, 1) if i == 0 else None))
            model.add(Dropout(hp.Float('dropout_' + str(i), 0.1, 0.5, step=0.1)))

        # 添加Dense輸出層
        model.add(Dense(1))

        # 自定義損失函數，忽略 NaN 值
        def custom_loss(y_true, y_pred):
            mask = tf.math.is_finite(y_true)  # 只計算非 NaN 的值
            return tf.reduce_mean(tf.square(
                tf.boolean_mask(y_true - y_pred, mask)))  # tf.reduce_mean(tf.square(...))：這部分計算非 NaN 值的均方誤差(MSE)。

        # 修改模型編譯部分，使用自定義損失
        model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss=custom_loss)

        return model

    def build_default_lstm(input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)

    data = json.loads(request.body)
    unique_id = data.get("unique_id")
    try:
        # 查找當前用戶的特定 model_name 的 auth_model 記錄
        if model_name:
            auth_model_entry = auth_model.objects.get(author=request.user, model_name=model_name)
            shutil.copyfile(os.path.join(settings.MEDIA_ROOT, 'auth_data', 'model_save', auth_model_entry.file_path),
                            os.path.join(settings.MEDIA_ROOT, 'model_save',
                                         f'{request.user.username}_{unique_id}_model_last.keras'))
            scaler_X = pickle.loads(auth_model_entry.scaler_X)
            scaler_y = pickle.loads(auth_model_entry.scaler_y)
            with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl'),
                      'wb') as file:
                pickle.dump(scaler_X, file)
            with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_y.pkl'),
                      'wb') as file:
                pickle.dump(scaler_y, file)
            print('讀取成功')
            return JsonResponse({'message': '讀取成功'})
    except auth_model.DoesNotExist:
        # 如果找不到符合條件的記錄，返回錯誤訊息
        return JsonResponse({'error': '指定的模型不存在或不屬於當前用戶。'}, status=404)
    # 讀取檔案並轉為 DataFrame
    input_data = data.get("origin_data")
    print('data type:', type(input_data))
    print('oringin_data', input_data)
    df = pd.DataFrame(input_data)
    # 設定索引，並選擇欄位
    if data.get('timing_data') == '按表格順序(時序選項)':
        df = df[[data.get('predict_data')]]
        df.index = range(1, len(df) + 1)
    else:
        timing_data = data.get('timing_data')
        df = df.set_index(timing_data)[[data.get('predict_data')]]

    # 創建訓練數據
    try:
        X_train = df.index.astype(int).values
        print('int')
    except:
        try:
            X_train = df.index.astype(float).values
            print('float')
        except:
            try:
                X_train = pd.to_datetime(df.index).astype(int).values
                print('time')

            except:
                return JsonResponse({'error': '輸入時序不合法'}, status=403)
    y_train = df[data.get('predict_data')].str.replace(r'[p,]', '', regex=True).astype(float).values

    # 標準化處理
    # 創建並應用 MinMaxScaler
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_oringin = X_train
    X_train = X_train.reshape((X_train.shape[0], 1, 1))

    # 對 X_train 和 y_train 進行標準化
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    if data.get('train_type') == 'restart':
        if data.get('model_type') == 'default':
            model = build_default_lstm((X_train.shape[1], 1))
            print('==============================================================default')
        elif data.get('model_type') == 'random':
            # 使用RandomSearch進行隨機搜索
            tuner = RandomSearch(
                build_random_lstm,
                objective='val_loss',
                max_trials=20,  # 增加搜索次數
                executions_per_trial=1,
                directory=tempfile.mkdtemp(),
                project_name='lstm_tuning',
            )

            tuner.search(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.2, verbose=1, callbacks=[])

            print('==========================================================')

            model = tuner.get_best_models()[0]
    elif data.get('train_type') == 'continue':
        model = tf.keras.models.load_model(
            os.path.join(settings.MEDIA_ROOT, 'model_save', f'{request.user.username}_{unique_id}_model_last.keras'))

    model.fit(X_train_scaled, y_train_scaled, epochs=int(data.get("epoch")), verbose=1)
    print('X_train_scaled_train:', X_train_scaled)
    # 預測
    predictions = model.predict(X_train_scaled)

    # 反轉標準化
    predictions = scaler_y.inverse_transform(predictions)

    # 將縮放器保存為 .pkl 文件
    with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl'),
              'wb') as file:
        pickle.dump(scaler_X, file)
    with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_y.pkl'),
              'wb') as file:
        pickle.dump(scaler_y, file)

    model.save(os.path.join(settings.MEDIA_ROOT, 'model_save', f'{request.user.username}_{unique_id}_model_last.keras'))

    # 生成圖片的 URL
    image_url = make_lstm_plot(X_train_oringin, X_train_oringin, y_train, predictions, request.user.username, unique_id,
                               True)

    # 回傳 JSON 包含圖片的 URL
    return JsonResponse({"predict_image_url": image_url})


def lstm_predict(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)
    data = json.loads(request.body)
    unique_id = data.get('unique_id')
    model = tf.keras.models.load_model(
        os.path.join(settings.MEDIA_ROOT, 'model_save', f'{request.user.username}_{unique_id}_model_last.keras'))
    with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    input_data = data.get("origin_data")
    df = pd.DataFrame(input_data)
    # 設定索引，並選擇欄位
    if data.get('timing_data') == '按表格順序(時序選項)':
        df = df[[data.get('predict_data')]]
        df.index = range(1, len(df) + 1)
    else:
        print('data.get(timing_data):', data.get('timing_data'))
        print('data.get(predict_data):', data.get('predict_data'))
        timing_data = data.get('timing_data')
        df = df.set_index(timing_data)[[data.get('predict_data')]]

    try:
        index_value = df.index.astype(int).values
        interval = index_value[-1] - index_value[-2]
        new_datas = [index_value[-1] + interval * i for i in range(1, int(data.get('predict_num')))]
        X_test = np.array(new_datas)
        print('int')
    except:
        try:

            index_value = df.index.astype(float).values
            interval = index_value[-1] - index_value[-2]
            new_datas = [index_value[-1] + interval * i for i in range(1, int(data.get('predict_num')))]
            X_test = np.array(new_datas)
            print('float')

        except:
            try:

                index_value = pd.to_datetime(df.index).values
                # 計算最後兩個資料之間的間隔
                interval = index_value[-1] - index_value[-2]

                # 生成後續的資料
                new_datas = [index_value[-1] + interval * i for i in range(1, int(data.get('predict_num')))]
                new_datas = np.array(new_datas, dtype='datetime64[ns]')

                # 合併原始資料和新資料
                X_test = np.array(new_datas)

                print('time')
            except:
                return JsonResponse({'error': '輸入時序不合法'}, status=403)
    x_true = index_value
    X_test_oringin = X_test
    X_test = X_test.reshape((X_test.shape[0], 1, 1))
    y_test = df[data.get('predict_data')].str.replace(r'[p,]', '', regex=True).astype(float).values

    # 對 X_train 和 y_train 進行標準化
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_train_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
    print('X_test_scaled_predict:', X_test_scaled)
    print('y_train_scaled_predict:', y_train_scaled)
    # 預測
    predictions = model.predict(X_test_scaled)

    # 反轉標準化
    predictions = scaler_y.inverse_transform(predictions)

    # 生成圖片的 URL
    image_url = make_lstm_plot(x_true, X_test_oringin, y_test, predictions, request.user.username, unique_id, True)

    # 回傳 JSON 包含圖片的 URL
    return JsonResponse({"predict_image_url": image_url})


def cnn_train(request, model_name=None):
    def create_default_cnn_model(input_shape, num_classes):
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

    def create_random_cnn_model(hp):
        model = Sequential()
        num_filters = hp.Choice('num_filters', values=[32, 64, 128])
        kernel_size = hp.Choice('kernel_size', values=[3, 5])
        activation = hp.Choice('activation', values=['relu', 'tanh'])
        dropout_rate = hp.Choice('dropout_rate', values=[0.3, 0.4, 0.5])
        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])

        model.add(Conv2D(num_filters, (kernel_size, kernel_size), activation=activation, input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(num_filters * 2, (kernel_size, kernel_size), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_image_data(train_dir, img_size, num_classes, batch_size=32):
        # 使用 ImageDataGenerator 來自動加載和預處理圖片
        img_size = img_size[:2]
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
            validation_split=0.2
        )

        # 创建一个图像数据生成器，仅对验证数据进行归一化处理
        test_datagne = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        # 訓練數據集
        train_generator = train_datagne.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical' if num_classes > 2 else 'binary',  # 對應多類分類問題
            subset='training'  # 使用 80% 圖片作為訓練數據
        )

        # 驗證數據集
        validation_generator = test_datagne.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical' if num_classes > 2 else 'binary',
            subset='validation'  # 使用 20% 圖片作為驗證數據
        )
        print('訓練類別', train_generator.class_indices)
        return train_generator, validation_generator

    def handle_uploaded_zip(file_obj, unique_id):
        # 構建解壓縮路徑
        extract_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', f'{request.user.username}_{unique_id}_test_dir')

        if os.path.exists(extract_dir):
            for root, dirs, files in os.walk(extract_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
        else:
            os.makedirs(extract_dir)

        # 直接使用 file_obj 進行解壓縮
        with zipfile.ZipFile(file_obj, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir  # 返回解壓後的資料夾路徑

    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)
    json_data = json.loads(request.POST['json_data'])
    unique_id = json_data['unique_id']

    # 查找當前用戶的特定 model_name 的 auth_model 記錄
    if model_name:
        auth_model_entry = auth_model.objects.get(author=request.user, model_name=model_name)
        shutil.copyfile(os.path.join(settings.MEDIA_ROOT, 'auth_data', 'model_save', auth_model_entry.file_path),
                        os.path.join(settings.MEDIA_ROOT, 'model_save',
                                     f'{request.user.username}_{unique_id}_model_last.keras'))

        with open(os.path.join(settings.MEDIA_ROOT, 'json',
                               f'{request.user.username}_{unique_id}_data_class_last.json'), 'w') as f:
            json.dump(auth_model_entry.data_class, f)
        return JsonResponse({'message': '讀取成功'})


    epoch = int(json_data['epoch'])
    num_classes = 2  # 类别數
    input_shape = (128, 128, 3)
    base_dir1 = handle_uploaded_zip(request.FILES['file'], unique_id)
    train_dir = os.path.join(base_dir1, os.listdir(base_dir1)[0])  # 使用者上傳的圖片數據夾路徑
    train_generator, validation_generator = load_image_data(train_dir, img_size=input_shape, num_classes=num_classes)
    if json_data['train_type'] == 'restart':
        if json_data['model_type'] == 'default':
            model = create_default_cnn_model(input_shape, num_classes)
        elif json_data['model_type'] == 'random':
            tuner = RandomSearch(
                create_random_cnn_model,
                objective='val_accuracy',
                max_trials=10,
                executions_per_trial=1,
                directory=tempfile.mkdtemp(),  # 使用臨時路徑
                project_name='cnn_tuning'
            )

            # 使用生成器來進行搜索
            tuner.search(
                train_generator,
                validation_data=validation_generator,
                epochs=epoch,
                steps_per_epoch=len(train_generator),
                validation_steps=len(validation_generator),
                verbose=2
            )

            # 獲取最佳模型，並根據需要進行額外訓練
            model = tuner.get_best_models(num_models=1)[0]


    elif json_data['train_type'] == 'continue':
        try:
            model = tf.keras.models.load_model(os.path.join(settings.MEDIA_ROOT, 'model_save',
                                                            f'{request.user.username}_{unique_id}_model_last.keras'))
        except:
            return JsonResponse({"error": "未找到訓練資料 請先進行訓練"}, status=403)

    else:
        return JsonResponse({"error": "未知的請求"}, status=403)

    if json_data['model_type'] == 'default':
        history = model.fit(train_generator, epochs=epoch, batch_size=32, validation_data=validation_generator,
                            verbose=2)
    elif json_data['model_type'] == 'random':
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10,  # 可以設定額外的 epochs
            steps_per_epoch=len(train_generator),
            validation_steps=len(validation_generator),
            verbose=2
        )
    model.save(os.path.join(settings.MEDIA_ROOT, 'model_save', f'{request.user.username}_{unique_id}_model_last.keras'))
    data_last_class_path = os.path.join(settings.MEDIA_ROOT, 'json',
                                        f'{request.user.username}_{unique_id}_data_class_last.json')

    with open(data_last_class_path, 'w') as f:
        json.dump(train_generator.class_indices, f)
    # 从训练历史中提取训练准确率和验证准确率
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training_accuracy')  # 训练准确率为蓝色线
    plt.plot(epochs, val_acc, 'r', label='Validation_accuracy')  # 验证准确率为红色线
    plt.title('Training and Validation accuracy')  # 设置图表标题
    plt.legend()  # 显示图例
    # 使用 MaxNLocator 自動控制 x 軸標籤數量
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=10))  # nbins 控制最多顯示的標籤數量

    plt.xticks(rotation=45, ha='right')  # 旋轉 x 軸標籤，避免擁擠
    plt.tight_layout()
    plot_path = os.path.join(settings.MEDIA_ROOT, 'plots')

    # 3. 圖片檔案名稱 (避免重名使用 timestamp)
    plt_filename = f"{request.user.username}_{unique_id}_keep_plot.png"
    plt_file_path = os.path.join(settings.MEDIA_ROOT, 'plots', plt_filename)

    # 4. 儲存圖片
    plt.savefig(plt_file_path)
    plt.close()

    # 5. 生成圖片的 URL
    image_url = f"{settings.MEDIA_URL}plots/{plt_filename}"

    # 6. 回傳 JSON 包含圖片的 URL
    return JsonResponse({"predict_image_url": image_url, "unique_id": unique_id})


def cnn_predict(request):
    def predict_image(file, model, img_size=(128, 128)):
        # 加載圖片並進行預處理
        img = image.load_img(BytesIO(file.read()), target_size=img_size)
        img_array = image.img_to_array(img) / 255.0  # 圖片標準化
        img_array = np.expand_dims(img_array, axis=0)  # 增加批次維度

        # 用模型進行預測
        predictions = model.predict(img_array)

        return predictions

    json_data = json.loads(request.POST['json_data'])
    unique_id = json_data['unique_id']
    cnn_last_model_dir = os.path.join(settings.MEDIA_ROOT, 'model_save',
                                      f'{request.user.username}_{unique_id}_model_last.keras')
    model = tf.keras.models.load_model(cnn_last_model_dir)  # 加載已訓練模型
    print(request.FILES)
    predictions = predict_image(request.FILES['image'], model)
    data_last_class_path = os.path.join(settings.MEDIA_ROOT, 'json',
                                        f'{request.user.username}_{unique_id}_data_class_last.json')
    with open(data_last_class_path, 'r') as f:
        class_names = list(json.load(f))
    print(class_names)
    print(predictions[0])
    print('預測結果為', class_names[int(predictions[0] > 0.5)])
    return JsonResponse({"predict_result": class_names[int(predictions[0] > 0.5)]})


def transformer_train(request):
    classifier = pipeline("text-classification", model="bert-base-chinese")

    def predict(texts):
        results = classifier(texts, truncation=True)
        predictions = [{"text": text, "label": result["label"], "score": result["score"]}
                       for text, result in zip(texts, results)]
        return predictions

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            file_content = data.get('fileContent')
            is_csv = data.get('isCsv')

            if not file_content:
                return JsonResponse({"error": "No file content provided"}, status=400)

            if is_csv:
                texts = file_content.split('\n')
            else:
                texts = [line.strip() for line in file_content.splitlines() if line.strip()]

            results = predict(texts)
            return JsonResponse({"results": results}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)


def save_model(request):
    def save_model_file(AuthModel, data, unique_id):
        cnn_last_model_dir = os.path.join(settings.MEDIA_ROOT, 'model_save',
                                          f'{request.user.username}_{unique_id}_model_last.keras')
        auth_model_file_path = os.path.join(settings.MEDIA_ROOT, 'auth_data', 'model_save',
                                            f'{request.user.username}_model_{data.get("model_name")}.keras')
        shutil.copyfile(cnn_last_model_dir, auth_model_file_path)
        AuthModel.keras_type = data.get("keras_type")
        plt_file_path = os.path.join(settings.MEDIA_ROOT, 'plots', f"{request.user.username}_{unique_id}_keep_plot.png")
        AuthModel.train_image.save(f'{data.get("model_name")}_{os.path.basename(plt_file_path)}',
                                   open(plt_file_path, 'rb'))
        data_last_class_path = os.path.join(settings.MEDIA_ROOT, 'json',
                                            f'{request.user.username}_{unique_id}_data_class_last.json')
        if os.path.exists(data_last_class_path):
            with open(data_last_class_path, 'r') as f:
                AuthModel.data_class = json.load(f)
        scaler_X_path = os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl')
        if data.get("keras_type") == 'Lstm':
            with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl'),
                      'rb') as f:
                scaler_X = pickle.load(f)
                AuthModel.scaler_X = pickle.dumps(scaler_X)
            with open(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_y.pkl'),
                      'rb') as f:
                scaler_y = pickle.load(f)
                AuthModel.scaler_y = pickle.dumps(scaler_y)
        AuthModel.save()

    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)
    data = json.loads(request.body)
    unique_id = data['unique_id']
    if auth_model.objects.filter(author=request.user, model_name=data.get("model_name")).exists() and not data.get(
            'check'):
        return JsonResponse({"message": "模型名稱重複 是否覆蓋"})
    elif data.get('check'):
        try:
            AuthModel = auth_model.objects.get(author=request.user, model_name=data.get("model_name"))
        except auth_model.DoesNotExist:
            return JsonResponse({"error": 'Model not found'}, status=404)
    else:
        AuthModel = auth_model(model_name=data.get("model_name"), author=request.user,
                               file_path=f'{request.user.username}_model_{data.get("model_name")}.keras')
    save_model_file(AuthModel, data, unique_id)

    return JsonResponse({}, status=204)


def show_auth_models(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)
    user = request.user
    auth_models = auth_model.objects.filter(author=user).values('model_name', 'train_image', 'updated_at', 'keras_type')
    auth_models_list = list(auth_models)
    return JsonResponse(auth_models_list, safe=False)


@csrf_exempt
def del_temporary_files(request):
    def remove_if_exist(file):
        try:
            os.remove(file)
            print('delete_file:', file)
        except FileNotFoundError:
            pass

    print(
        'del_temporary_files執行===========================================================================================')
    if not request.user.is_authenticated:
        return JsonResponse({"error": "用户未登录，请先登录。"}, status=403)
    unique_id = json.loads(request.body)['unique_id']
    remove_if_exist(
        os.path.join(settings.MEDIA_ROOT, 'json', f'{request.user.username}_{unique_id}_data_class_last.json'))

    remove_if_exist(os.path.join(settings.MEDIA_ROOT, 'plots', f"{request.user.username}_{unique_id}_keep_plot.png"))
    remove_if_exist(
        os.path.join(settings.MEDIA_ROOT, 'model_save', f'{request.user.username}_{unique_id}_model_last.keras'))

    remove_if_exist(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_X.pkl'))
    remove_if_exist(os.path.join(settings.MEDIA_ROOT, 'pkl', f'{request.user.username}_{unique_id}_scaler_y.pkl'))

    return JsonResponse({}, status=204)


def get_unique_id(request):
    unique_id = str(uuid.uuid4())
    return JsonResponse({'unique_id': unique_id})
