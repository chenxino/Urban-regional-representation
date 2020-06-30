from django.urls import path
from . import views
from django.conf.urls import url

app_name = 'pictures'
urlpatterns = [
    path('', views.upload, name='upload'),
    path('extraction/action/', views.codeprocess, name='codeprocess'),
    path('extraction/<str:code>/', views.extraction, name='extraction'),
    url(r'download/(?P<file_name>.*)/$', views.download_file),
]
