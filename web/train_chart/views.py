from django.shortcuts import render

# Create your views here.
def train(request):
    return render(request, 'train_chart/train.html')

def introduction(request):
    return render(request, 'train_chart/introduction.html')

def vaild(request):
    return render(request, 'train_chart/val.html')