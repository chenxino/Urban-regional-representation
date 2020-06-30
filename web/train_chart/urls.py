from django.urls import path

from . import views

app_name = 'train_chart'
urlpatterns = [

    path('',views.introduction, name='introduction'),
    path('train', views.train, name='train'),
    path('vaild',views.vaild,name='vaild'),
]