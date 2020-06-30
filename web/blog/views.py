from django.shortcuts import render, get_object_or_404
from .models import Post
from django.http import HttpResponse


# Create your views here.
def index(request):
    post_list = Post.objects.all().order_by('-created_time')
    return render(request, 'blog/index.html', context={'post_list': post_list})
def detail(request,pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'blog/detail.html', context={'post': post})
def map(request):
    return render(request, 'blog/map.html')
def map_id(request, id):
    id = 'blog/map1/map'+str(id)+'.html'
    return render(request, id)
def poi(request):
    return render(request, 'blog/poi.html')
def taxi(request):
    return render(request, 'blog/taxi.html')
