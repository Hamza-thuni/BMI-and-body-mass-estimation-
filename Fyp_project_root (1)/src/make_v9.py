
with open('live_bmi_demo_v8_pi.py', 'r') as f:
    code = f.read()

code = code.replace('v8_pi', 'v9')
code = code.replace('V8', 'V9')
code = code.replace('v8_physical', 'v9_hybrid')
code = code.replace('models_dir(\'v8\')', 'models_dir(\'v9\')')
code = code.replace('hybrid_v8', 'hybrid_v9')

# Remove PCA processing lines
code = code.replace('Xp = bundle[\'pca\'].transform(Xs)', 'Xp = Xs')

with open('live_bmi_demo_v9.py', 'w') as f:
    f.write(code)

print('V9 script created successfully.')

