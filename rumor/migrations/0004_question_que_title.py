# Generated by Django 3.1.7 on 2021-04-07 04:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rumor', '0003_auto_20210406_1505'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='que_title',
            field=models.CharField(default='具体内容点击详情进行查看！', max_length=200),
        ),
    ]
