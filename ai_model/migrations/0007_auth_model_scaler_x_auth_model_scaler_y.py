# Generated by Django 5.1.1 on 2024-11-17 17:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai_model', '0006_auth_model_keras_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='auth_model',
            name='scaler_x',
            field=models.BinaryField(default=b''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='auth_model',
            name='scaler_y',
            field=models.BinaryField(default=b''),
            preserve_default=False,
        ),
    ]
