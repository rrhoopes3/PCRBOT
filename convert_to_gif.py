from moviepy.editor import VideoFileClip

# Convert bearish.mp4 to GIF
clip = VideoFileClip("bearish.mp4")
clip.write_gif("bearish.gif")

# Convert bullish.mp4 to GIF
clip = VideoFileClip("bullish.mp4")
clip.write_gif("bullish.gif")