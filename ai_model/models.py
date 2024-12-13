# models.py
from django.db import models
from django.conf import settings

class auth_model(models.Model):
    model_name = models.CharField(max_length=100)
    keras_type = models.CharField(max_length=100)
    train_image = models.ImageField(upload_to='images/')
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=255,default='/default/path/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  # 每次更新都自動設置為當前時間
    data_class = models.JSONField(default=list)
    scaler_X = models.BinaryField()  # 确保默认值符合类型
    scaler_y = models.BinaryField()
