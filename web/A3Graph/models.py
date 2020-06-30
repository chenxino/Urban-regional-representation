from django.db import models
from login.models import User
# Create your models here.
class Document(models.Model):
    code = models.CharField(max_length=8, default='code')
    author = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    def __str__(self):
        return self.code
