from django.shortcuts import render,render_to_response
from pictures.forms import UserForm, DocumentForm
from django.http import HttpResponse
from pictures.models import RawImg
from pictures.process import ImgProcess
from random import Random
import os
from PIL import Image
# Create your views here.
from django.http import StreamingHttpResponse,FileResponse


def random_str(randomlength=4):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def image_process(file, code, i):
    file_name = str(i)+ '.png'
    file_path = os.path.join('media/out', code, file_name).replace('\\', '/')
    fromImg = Image.open(file)
    fromImg.save(file_path, quality=95)


def upload(request):
    if request.method == 'POST':
        uf = UserForm(request.POST,request.FILES)

        if uf.is_valid():
            # 创建提取码
            flag = 0
            while(flag==0):
                code = random_str()
                if RawImg.objects.filter(code__startswith=code):
                    flag = 0
                else:
                    flag = 1
            new_dir = 'media/out/'+code
            os.makedirs(new_dir)

            # 从表单获取图像数据
            img = uf.cleaned_data['img_01']

            # for i in range(1,1):
            #     if i<10 :
            #         s_str = 'img_0'+str(i)
            #     else:
            #         s_str = 'img_'+str(i)
            #     img.append(uf.cleaned_data[s_str])

            # 图像处理
            # for i in range(1,2):
            #     image_process(img[i-1], code, i)
            image_process(img, code, 1)
            ImgProcess(code)
            # 记录提取码
            user = RawImg()
            user.code = code

            if request.session.get('is_login', None):
                user.author_id = request.session['user_id']

            user.save()

            return render(request,'pictures/show.html',{'code':code})
    else:
        uf = UserForm()
    return render(request,'pictures/index1.html',{'uf':uf})


def extraction(request, code):
    if code == '0':
        return render(request, 'pictures/extraction.html', {'code': code})
    else:
        if RawImg.objects.filter(code__startswith=code):
            return render(request, 'pictures/show.html', {'code': code})
        else:
            return render(request, 'pictures/404.html')


def codeprocess(request):
    code = request.POST.get('code')
    return render_to_response('pictures/show.html', {'code': code})

def download_file(request,file_name):

    the_file_name = 'E:/webapp/blogproject/media/out/'+ file_name
    print(the_file_name,'1111111111111111111111')
    file = open(the_file_name, 'rb')
    response = FileResponse(file)

    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachement;filename="{0}"'.format(file_name)
    return response