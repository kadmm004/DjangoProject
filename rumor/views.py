import datetime
import uuid

import numpy as np
import paddle
import oss2
import os

import pytz
from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse
# coding=utf-8
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from paddle import fluid
from paddle.fluid.dygraph import to_variable

from WebDemo.settings import BASE_DIR
from rumor import models
from rumor.cnn_method import train_parameters, CNN
from rumor.models import User, IMG, News, get_userid, Question, Answer, Like

# auth = oss2.Auth('my-own-key', 'my-own-key')
endpoint = 'https://oss-cn-chengdu.aliyuncs.com'
# bucket域名：
# xdmeng-rumor-demo.oss-cn-chengdu.aliyuncs.com
bucket = oss2.Bucket(auth, endpoint, 'my-own-lab')
# 基础访问链接，根据情况修改
base_url = 'my-own-url'


def toLogin(request):
    return render(request, 'Login.html')


def toRegister(request):
    return render(request, 'Register.html')


def toDetect(request):
    return render(request, 'detect.html')


def toQuestion(request):
    ques = Question.objects.all()
    context = {
        'ques': ques,
    }
    return render(request, 'Question.html', context=context)


def toMyinfo(request):
    username = request.session.get("username")
    u = User.objects.filter(username=username).first()
    context = {
        'u': u,
        'u_img': u.img_url,
    }

    return render(request, 'Myinfo.html', context)


def toNewsitem(request):
    news_id = request.GET.get('news_id')
    request.session['news_id'] = news_id
    new = News.objects.filter(news_id=news_id).first()
    # 读取新闻内容
    new_dir = new.news_content
    f = open(new_dir, encoding='utf-8', newline='\r')
    # 行读取
    res = ""
    for line in f.readlines():
        # line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        res = res+line+"\n"
    # line = f.readlines()
    f_content = f.read()
    f.close()
    context = {'f_content': f_content,
               'new': new,
               'res': res
               }
    return render(request, 'NewsItem.html', context)


def likeNews(request):
    news_id = request.session.get('news_id')
    new = News.objects.filter(news_id=news_id).first()
    # 读取新闻内容
    new_dir = new.news_content
    f = open(new_dir, encoding='utf-8')
    f_content = f.read()
    f.close()
    user_id = request.session.get("user_id")
    u = User.objects.filter(user_id=user_id).first()
    like = Like(news_id=new, user_id=u)
    like.save()
    context = {'f_content': f_content,
               'new': new
               }
    # return render(request, 'NewsItem.html', context)
    return redirect('/rumor/newitem/?news_id='+news_id)


def toAskque(request):
    return render(request, 'askque.html')


def toQueInfo(request):
    que_id = request.GET.get('que_id')
    request.session['que_id'] = que_id
    que = Question.objects.filter(que_id=que_id).first()
    ans = Answer.objects.filter(que_id=que).all()
    # messages.success(request, que.user_id.email)
    context = {
        'que': que,
        'ans': ans
    }
    return render(request, 'QueInfo.html', context)


def Login_view(request):
    # 从form表单中的username获取值，如果找不到就给一个null
    username = request.POST.get("username", "")
    password = request.POST.get("password", "")

    if username and password:
        c = User.objects.filter(username=username, password=password).count()
        u = User.objects.filter(username=username, password=password).first()
        if c >= 1:
            request.session['username'] = username
            request.session['user_id'] = get_userid(u)
            request.session['state'] = "登出"

            # return render(request, 'index.html')
            return redirect('/rumor/index/')
        else:
            return HttpResponse("帐号密码错误！")
    else:
        return HttpResponse("帐号密码不完整！")


def Register_view(request):
    username = request.POST.get("username", "")
    password = request.POST.get("password", "")
    password_ack = request.POST.get("password_ack", "")
    nickname = request.POST.get("nickname", "")
    email = request.POST.get("email", "")

    count = User.objects.filter(username=username).count()
    if count:
        messages.error(request, "用户名重复，请重新输入！")
        return render(request, 'Register.html')
    else:
        if username and password and password_ack:
            # 用户名是唯一的，虽然models这么设置了但是需要有报错提示
            if password_ack == password:
                # 头像上传
                image = request.FILES.get("myfile").read()
                # 通过上面封装的方法把文件上传
                image_url = uploadImg(image)
                if nickname:
                    user = User(username=username, password=password, nickname=nickname, email=email, img_url=image_url)
                else:
                    user = User(username=username, password=password, nickname=username, email=email, img_url=image_url)

                user.save()
                request.session['username'] = username
                request.session['user_id'] = get_userid(user)
                request.session['state'] = "登出"
                messages.success(request, "注册成功！")
                return redirect('/rumor/index/')
            else:
                messages.error(request, "两次密码不一致，请重新输入！")
                return render(request, 'Register.html')
        else:
            messages.error(request, "请输入完整的帐号和密码！")
            return render(request, 'Register.html')


def Index_view(request):
    news = News.objects.all()
    context = {
        'news': news,
    }
    for i in news:
        print(i.new_img.url)
    return render(request, 'index.html', context=context)


def pwdEdit(request):
    username = request.session.get("username")
    u = User.objects.filter(username=username).first()
    ack_email = request.POST.get("ack_email", "")
    old_pwd = request.POST.get("old_pwd", "")
    new_pwd = request.POST.get("new_pwd", "")
    ack_pwd = request.POST.get("ack_pwd", "")

    context = {
        'u': u,
        'u_img': u.img_url,
    }

    if ack_email == u.email:
        if old_pwd == u.password:
            if new_pwd == ack_pwd:
                # models.User.objects.filter(username='username').update(password=new_pwd)
                u.password = new_pwd
                u.save()
                print(u.password)
                messages.success(request, "修改成功！")
                return render(request, 'Myinfo.html', context)
            else:
                messages.success(request, "两次密码不一致，请重新输入！")
                return render(request, 'Myinfo.html', context)
        else:
            messages.success(request, "旧密码输入错误！")
            return render(request, 'Myinfo.html', context)
        return
    else:
        messages.success(request, "邮箱输入错误！")
        return render(request, 'Myinfo.html', context)


def infoEdit(request):
    username = request.session.get("username")
    u = User.objects.filter(username=username).first()
    old_email = u.email
    context = {
        'u': u,
        'u_img': u.img_url,
    }
    new_nickname = request.POST.get("new_nickname", username)
    new_email = request.POST.get("new_email", old_email)
    # 头像上传
    image = request.FILES.get("myfile").read()
    if not image:
        messages.success(request, "上传失败")
        return render(request, 'Myinfo.html', context)
    # 通过上面封装的方法把文件上传
    image_url = uploadImg(image)
    u.nickname = new_nickname
    u.email = new_email
    u.img_url = image_url
    u.save()
    context = {
        'u': u,
        'u_img': u.img_url,
    }
    messages.success(request, "修改成功！")
    return render(request, 'Myinfo.html', context)


def askQue(request):
    title = request.POST.get("ask_title", "")
    content = request.POST.get("ask_content", "")
    user_id = request.session.get("user_id")
    u = User.objects.filter(user_id=user_id).first()
    # 返回时间格式的字符串
    ask_time = datetime.datetime.now().strftime('%Y-%m-%d')
    que = Question(que_title=title, que_content=content, ask_time=ask_time, user_id=u)
    que.save()
    messages.success(request, "提问发表成功！")
    return redirect('/rumor/toQuestion/')


def uploadImg(image):
    # 生成文件编号，如果文件名重复的话在oss中会覆盖之前的文件
    number = uuid.uuid4()
    # 生成文件名
    base_img_name = str(number) + '.jpg'
    # 生成外网访问的文件路径
    image_name = base_url + base_img_name
    # 这个是阿里提供的SDK方法 bucket是调用的4.1中配置的变量名
    res = bucket.put_object(base_img_name, image)
    # 如果上传状态是200 代表成功 返回文件外网访问路径
    # 下面代码根据自己的需求写
    if res.status == 200:
        return image_name
    else:
        return HttpResponse('false')


def toTest(request):
    return render(request, 'test.html')


def replyQue(request):
    que_id = request.session.get('que_id')
    que = Question.objects.filter(que_id=que_id).first()
    # 返回时间格式的字符串
    ans_time = datetime.datetime.now().strftime('%Y-%m-%d')
    ans_content = request.POST.get("ans_content", "暂无内容")
    user_id = request.session.get("user_id")
    u = User.objects.filter(user_id=user_id).first()
    ans = Answer(ans_content=ans_content, ans_time=ans_time, que_id=que, user_id=u)
    ans.save()

    ans = Answer.objects.filter(que_id=que).all()
    context = {
        'que': que,
        'ans': ans
    }

    return render(request, 'QueInfo.html', context)


def toMyLike(request):
    user_id = request.session.get("user_id")
    u = User.objects.filter(user_id=user_id).first()
    like = Like.objects.filter(user_id=u).all()
    context = {
        'like': like,
    }
    return render(request, 'MyLike.html', context)


def toMyAsk(request):
    user_id = request.session.get("user_id")
    u = User.objects.filter(user_id=user_id).first()
    que = Question.objects.filter(user_id=u).all()
    context = {
        'que': que,
    }
    return render(request, 'MyAsk.html', context)


@csrf_exempt
def test(request):
    print(request.POST.get('test'))
    # 获取前端ajax传的文件 使用read()读取b字节文件
    image = request.FILES.get('image').read()
    # 通过上面封装的方法把文件上传
    image_url = uploadImg(image)
    # 这里没有做判断验证只是测试代码 根据自己的需求需要判断
    return image_url


def show_img(request):
    return render(request, 'test.html')


paddle.enable_static()
# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
# 这句话会抛警告
# infer_exe.run(fluid.default_startup_program())

save_path = 'D:/study_mxd/cnn_rumor/work/infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=infer_exe)


# lstm方法获取数据
def get_data(sentence):
    # 读取数据字典
    with open('D:/study_mxd/cnn_rumor/data/dict.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(np.array(dict_txt[s]).astype('int64'))
    return data


# cnn方法获取数据
def load_data(sentence):
    # 读取数据字典
    with open('D:/study_mxd/cnn_rumor/data/dict.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data


# cnn分类器
train_parameters["batch_size"] = 1
lab = ['谣言', '非谣言']

'''
# view方法实现_cnn
def Detect_view(request):
    text_ori = request.POST.get('detect_content')
    with fluid.dygraph.guard(place=fluid.CPUPlace()):
        data = load_data(text_ori)
        data_np = np.array(data)
        data_np = np.array(
            np.pad(data_np, (0, 150 - len(data_np)), "constant",
                   constant_values=train_parameters["vocab_size"])).astype(
            'int64').reshape(-1)

        infer_np_doc = to_variable(data_np)

        model_infer = CNN()
        model, _ = fluid.load_dygraph("D:/study_mxd/cnn_rumor/data/save_dir_1100.pdparams")
        model_infer.load_dict(model)
        model_infer.eval()

        result = model_infer(infer_np_doc)
        print('预测结果为：', lab[np.argmax(result.numpy())])
    content = {
        'text': text_ori,
        'res': lab[np.argmax(result.numpy())],
    }
    return render(request, 'detect.html', content)
'''


def Detect_view(request):
    text_ori = request.POST.get('detect_content')
    data = []
    # 获取图片数据
    data1 = get_data(text_ori)
    data.append(data1)

    # 获取每句话的单词数量
    base_shape = [[len(c) for c in data]]

    # 生成预测数据
    tensor_words = fluid.create_lod_tensor(data, base_shape, place)

    # 执行预测
    result = infer_exe.run(program=infer_program,
                           feed={feeded_var_names[0]: tensor_words},
                           fetch_list=target_var)

    # 分类名称
    names = ['谣言', '非谣言']

    # 获取结果概率最大的label
    for i in range(len(data)):
        lab = np.argsort(result)[0][i][-1]
        print('预测结果标签为：%d， 分类为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))
    content = {
        'text': text_ori,
        'lab': lab,
        'classification': names[lab],
        'res': result[0][0][lab],
    }
    return render(request, 'detect.html', content)

