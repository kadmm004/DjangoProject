# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '2.0.0'
major           = '2'
minor           = '0'
patch           = '0'
rc              = '0'
istaged         = True
commit          = 'fd9d6fdac285ba34f0e9a91c53bc07859ae2dd1c'
with_mkl        = 'ON'

def show():
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)

def mkl():
    return with_mkl
