from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('auth/', include('user.urls')),
    path('captcha/', include('captcha.urls')),  # 添加這一行來包括驗證碼路由
    path('ai_model/',include('ai_model.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)