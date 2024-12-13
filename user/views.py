from django.shortcuts import render, redirect
from django.contrib import messages
from user.forms import UserRegistrationForm
from django.http import JsonResponse
from captcha.models import CaptchaStore
from captcha.helpers import captcha_image_url
from django.core.mail import send_mail
from django.utils import timezone
import random
import json
from django.contrib.auth import authenticate, login , logout
from django.views.decorators.csrf import ensure_csrf_cookie


def generate_verification_code():
    """生成6位數字驗證碼"""
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])



def send_verification_code(request):
    """發送驗證碼到用戶的電子郵件並將驗證碼存儲在 session 中"""
    print('0')
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # 獲取 JSON 請求數據
            email = data.get('email')
        except json.JSONDecodeError:
            print('1')
            return JsonResponse({'error': '無效的 JSON 數據'}, status=400)

        if not email:
            print('1')
            return JsonResponse({'error': '缺少 email 參數'}, status=400)

        code = generate_verification_code()

        # 將驗證碼和過期時間保存到 session 中
        request.session['email_verification_code'] = code
        request.session['email_verification_expiration'] = (timezone.now() + timezone.timedelta(minutes=10)).strftime(
            '%Y-%m-%d %H:%M:%S')

        # 發送郵件
        send_mail(
            '您的郵件驗證碼',
            f'您的郵件驗證碼是 {code}，有效期為10分鐘。',
            'abcdefghijklm12311@gmail.com',  # 替換為你的發件人郵箱
            [email],
            fail_silently=False,
        )
        print('2')
        return JsonResponse({'message': '驗證碼已發送'})

    print('3')
    return JsonResponse({'error': '無效的請求方式'}, status=405)



def register(request):
    print('a')
    if request.method == 'POST':
        print('b')
        data = json.loads(request.body)
        form = UserRegistrationForm(data)
        if form.is_valid():
            # 獲取 session 中的驗證碼和過期時間
            session_code = request.session.get('email_verification_code')
            expiration_time_str = request.session.get('email_verification_expiration')
            expiration_time = timezone.datetime.strptime(expiration_time_str, '%Y-%m-%d %H:%M:%S')

            # 驗證驗證碼
            email_verification_code = form.cleaned_data.get("captcha")
            print('0')
            if timezone.now().replace(tzinfo=None) > expiration_time:
                print('message: 驗證碼已過期')
                return JsonResponse({'message': '驗證碼已過期'})
            elif email_verification_code != session_code:
                print('message: 驗證碼不正確')
                return JsonResponse({'message': '驗證碼不正確'})
            else:
                # 保存新用戶，不保存密碼原文
                user = form.save(commit=False)
                user.set_password(form.cleaned_data['password'])
                user.save()
                messages.success(request, '註冊成功，請登錄！')
                return JsonResponse({'message': '註冊成功，請登錄！','redirect_url': 'http://localhost:5173/login'})
        else:
            print(form.errors)
    else:
        JsonResponse({'message': '請求非POST'})

    return JsonResponse({'message': '註冊成功'})

def user_login(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # 獲取表單提交的用戶名、密碼和驗證碼
        email = data.get('email')
        password = data.get('password')
        captcha_key = data.get('captcha_key')
        captcha_value = data.get('captcha_value')

        # 查找驗證碼
        try:
            captcha = CaptchaStore.objects.get(hashkey=captcha_key)
        except CaptchaStore.DoesNotExist:
            messages.error(request, '驗證碼無效')
            return JsonResponse({'message': '驗證碼無效'})

        # 檢查驗證碼是否正確
        if captcha.response.lower() != captcha_value.lower():
            print('captcha.response:', captcha.response,'captcha_value:',captcha_value )
            messages.error(request, '驗證碼不正確')
            return JsonResponse({'message': '驗證碼不正確'})

        # 認證用戶
        user = authenticate(request, email=email, password=password)

        if user is not None:
            # 用戶認證成功，執行登入操作
            login(request, user)
            messages.success(request, '登入成功！')
            print('登入成功！')
            return JsonResponse({'message': '登入成功','redirect_url':'http://localhost:5173/'})  # 可重定向到首頁或其他頁面
        else:
            # 用戶認證失敗，提示錯誤消息
            messages.error(request, '無效的信箱或密碼！')
            return JsonResponse({'message': '無效的信箱或密碼！'})

    return JsonResponse({'message': '請求方法無效'})

def user_logout(request):
    if request.method == 'POST':
        # 執行登出操作
        logout(request)
        return JsonResponse({'message': '已登出', 'redirect_url': 'http://localhost:5173/'})  # 返回登出成功消息及重定向的URL
    else:
        return JsonResponse({'message': '請求方式無效'}, status=405)

@ensure_csrf_cookie
def get_user_info(request):
    print(request.user.is_authenticated)
    if request.user.is_authenticated:
        return JsonResponse({'islogin': True,'username': request.user.username})
    else:
        return JsonResponse({'islogin': False})

def get_captcha(request):
    new_key = CaptchaStore.generate_key()
    image_url = captcha_image_url(new_key)
    print('123456')
    return JsonResponse({
        'captcha_key': new_key,
        'captcha_image_url': image_url
    })

