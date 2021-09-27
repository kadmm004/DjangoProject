from django.conf import settings
from django.conf.urls.static import static

from django.urls import path

from rumor import views

urlpatterns = [
    path('', views.toLogin),
    # 登录
    path('toLogin', views.toLogin, name="toLogin"),
    path('Login/', views.Login_view),
    # 注册
    path('toRegister', views.toRegister, name="toRegister"),
    path('register/', views.Register_view),
    # 检测
    path('toDetect', views.toDetect, name="toDetect"),
    path('detect/', views.Detect_view, name="detect"),
    # 个人信息
    path('toMyinfo/', views.toMyinfo),
    # 提问，新闻列表
    path('toAskque', views.toAskque),
    # 问题详情页面
    path('toQueInfo/', views.toQueInfo),
    # 用户发表评论回复
    path('replyQue/', views.replyQue),
    # path('toQueInfo', views.toQueInfo),
    # 发布问题
    path('toQuestion/', views.toQuestion),
    path('askQue/', views.askQue),
    # 主页，新闻列表
    path('index/', views.Index_view),
    # 新闻详情
    path('newitem/', views.toNewsitem),
    # 新闻收藏
    path('likeNews/',views.likeNews),
    # 修改个人密码，信息
    path('pwdEdit/', views.pwdEdit),
    path('infoEdit/', views.infoEdit),
    # 跳转个人收藏
    path('toMyLike/', views.toMyLike),
    # 跳转个人提问
    path('toMyAsk/', views.toMyAsk),

    # 测试上传头像
    path('test/', views.toTest, name="test"),
    path('show/', views.show_img, name="show"),
    path('test', views.test, name='test'),


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
