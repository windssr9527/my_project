from django.urls import path, include
from . import views

app_name = 'ai_model'

urlpatterns = [
    path('lstm_train', views.lstm_train, name='lstm_train'),
    path('lstm_train/<str:model_name>/', views.lstm_train, name='lstm_train'),
    path('lstm_predict', views.lstm_predict, name='lstm_predict'),
    path('cnn_train', views.cnn_train, name='cnn_train'),
    path('cnn_train/<str:model_name>/', views.cnn_train, name='cnn_train'),
    path('cnn_predict', views.cnn_predict, name='cnn_predict'),
    path('save_model',views.save_model, name='save_model'),
    path('show_auth_models',views.show_auth_models,name='show_auth_models'),
    path('del_temporary_files',views.del_temporary_files,name='del_temporary_files'),
    path('get_unique_id',views.get_unique_id,name='get_unique_id'),

]
