# Use the official lightweight Python image.
FROM python:3.12-slim

# Copy local code to the container image.
WORKDIR /app
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH 

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 

