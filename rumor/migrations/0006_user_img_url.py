# Generated by Django 3.1.7 on 2021-04-12 03:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rumor', '0005_auto_20210410_1459'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='img_url',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
