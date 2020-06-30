from django.urls import path
from . import views
from django.conf.urls import url

app_name = 'A3Graph'

urlpatterns = [
    path('', views.list, name='list'),
    url(r'download/(?P<file_name>.*)/$', views.download_file),
    path('extraction/action/', views.codeprocess, name='codeprocess'),
    path('extraction/<str:code>/', views.extraction, name='extraction'),

]
