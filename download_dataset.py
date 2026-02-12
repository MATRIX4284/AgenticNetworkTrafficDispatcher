import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/comcast-telecom-complaints")

print("Path to dataset files:", path)