import os

os.system("luxai-s3 rule_based_jax/random/main.py rule_based_jax/random/main.py -o random.html")
os.system("luxai-s3 rule_based_jax/naive/main.py rule_based_jax/naive/main.py -o naive.html")
os.system("luxai-s3 jax_main.py jax_main.py -o purejaxrl.html")