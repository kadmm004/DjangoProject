from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.db import models


# Create your models here.
class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    # blank=True表示允许为空，null=True表示允许值是null
    username = models.CharField(unique=True, max_length=45)
    password = models.CharField(max_length=30)
    nickname = models.CharField(max_length=50, null=True, blank=True)
    email = models.CharField(max_length=30, blank=True, null=True)
    head_img = models.ImageField(upload_to='avatar', null=True, blank=True)
    img_url = models.CharField(max_length=500, null=True, blank=True)


class Question(models.Model):
    que_id = models.AutoField(primary_key=True)
    # username = models.ForeignKey('User', to_field='username', on_delete=models.CASCADE)
    user_id = models.ForeignKey('User', to_field='user_id', on_delete=models.CASCADE)
    que_content = models.TextField()
    que_title = models.CharField(max_length=200, default='具体内容点击详情进行查看！')
    ask_time = models.DateField(auto_now_add=True)


class News(models.Model):
    news_id = models.AutoField(primary_key=True)
    news_title = models.CharField(max_length=300, null=True, blank=True)
    news_content = models.CharField(max_length=100, null=True, blank=True)
    news_sum = models.CharField(max_length=100, null=True, blank=True)
    up_time = models.DateField(auto_now_add=True)
    new_img = models.ImageField(upload_to='news_img', null=True, blank=True)
    isTrue = models.BooleanField(default=False)


class Answer(models.Model):
    ans_id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey('User', to_field='user_id', on_delete=models.CASCADE)
    que_id = models.ForeignKey('Question', to_field='que_id', on_delete=models.CASCADE)
    ans_content = models.CharField(max_length=300)
    ans_time = models.DateField(auto_now_add=True)


class Like(models.Model):
    like_id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey('User', to_field='user_id', on_delete=models.CASCADE)
    news_id = models.ForeignKey('News', to_field='news_id', on_delete=models.CASCADE)


class IMG(models.Model):
    img = models.ImageField(upload_to='img')
    name = models.CharField(max_length=20)


def get_userid(self):
    return User.objects.get(username=self.username).user_id


def get_username(self):
    return User.objects.get(user_id=self.user_id).username
