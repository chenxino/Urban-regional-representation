from django.shortcuts import render,render_to_response
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import StreamingHttpResponse,FileResponse

from random import Random
import os
import numpy as np
from A3Graph.forms import DocumentForm
from A3Graph.models import Document
from A3Graph.A3 import main,read_feature2,four_ord_ppmi,tfidf_feature,output_features_mat

def random_str(randomlength=4):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def file2(file1, file2, code):
    destination = open(os.path.join('media/out2', code, 'feature.txt').replace('\\', '/'),'wb+')
    for chunk in file1:
        destination.write(chunk)
    destination.close()
    destination2 = open(os.path.join('media/out2', code, 'structure.txt').replace('\\', '/'), 'wb+')
    for chunk in file2:
        destination2.write(chunk)
    destination2.close()

def process(code):
    input_feature_mat = read_feature2(r"media/out2/%s/feature.txt" % code)
    print(input_feature_mat)
    input_feature_mat = tfidf_feature(input_feature_mat)
    c4 = np.loadtxt(r"media/out2/%s/structure.txt" % code)
    input_mat = four_ord_ppmi(c4, input_feature_mat.shape[0])
    output_feature_mat = output_features_mat(input_mat, input_feature_mat)
    print(output_feature_mat)
    ee = main(input_feature_mat, output_feature_mat)
    ee = ee.detach().numpy()
    np.savetxt("media/out2/%s/feature_A3.txt" % code, ee)

# Create your views here.
def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            # 创建提取码
            flag = 0
            while (flag == 0):
                code = random_str()
                if Document.objects.filter(code__startswith=code):
                    flag = 0
                else:
                    flag = 1
            new_dir = 'media/out2/' + code
            os.makedirs(new_dir)
            feature = form.cleaned_data['docfile1']
            structure = form.cleaned_data['docfile2']
            file2(feature, structure, code)

            process(code)


            user = Document(code=code)
            if request.session.get('is_login',None):
                user.author_id = request.session['user_id']
            user.save()
            return render(request, 'A3Graph/show.html', {'code': code})

            # # Redirect to the document list after POST
            # return HttpResponseRedirect(reverse('list'))
    else:
        form = DocumentForm()  # A empty, unbound form


    # Render list page with the documents and the form
    return render(request, 'A3Graph/list.html', {'form': form})


def download_file(request,file_name):
    the_file_name = 'E:/webapp/blogproject/media/out2/' + file_name
    print(the_file_name, '1111111111111111111111')
    file = open(the_file_name, 'rb')
    response = FileResponse(file)

    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachement;filename="{0}"'.format(file_name)
    return response


def extraction(request, code):
    if code == '0':
        return render(request, 'A3Graph/extraction.html', {'code': code})
    else:
        if Document.objects.filter(code__startswith=code):
            return render(request, 'A3Graph/show.html', {'code': code})
        else:
            return render(request, 'A3Graph/404.html')


def codeprocess(request):
    code = request.POST.get('code')
    return render_to_response('A3Graph/show.html', {'code': code})