export const secrets = {
  AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID || "YOUR_AWS_ACCESS_KEY_ID",
  AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY || "YOUR_AWS_SECRET_ACCESS_KEY",
  MLFLOW_TRACKING_URI: process.env.MLFLOW_TRACKING_URI || "http://localhost:5000",
}

export const env = {
  PYTHON_VERSION: process.env.PYTHON_VERSION || "3.9",
}
