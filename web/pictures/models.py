from django.db import models
from login.models import User

# Create your models here.

class RawImg(models.Model):
    code = models.CharField(max_length=8,default='code')
    author = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    # img_01 = models.ImageField(upload_to='upload',default='')
#    img_02 = models.ImageField(upload_to='out',default='')

    def __str__(self):
        return self.code


# Create your models here.
#
# class Post_img(models.Model):
#     code = models.CharField(max_length=200, default='code')
#
#     cover = models.ImageField(upload_to="images/", default=None, blank=True)
#     def __str__(self):
#         return self.code
#
# class Operation(models.Model):
#     created = models.DateTimeField(auto_now_add=True)
#     type = models.CharField(max_length=200)
#     post = models.ForeignKey(Post_img, on_delete=models.CASCADE, default=None)
#
