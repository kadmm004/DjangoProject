# Generated by Django 3.1.7 on 2021-04-06 07:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rumor', '0002_img'),
    ]

    operations = [
        migrations.AddField(
            model_name='news',
            name='news_title',
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
        migrations.AlterField(
            model_name='news',
            name='news_content',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
