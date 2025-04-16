import matplotlib.pyplot as plt
import matplotlib
import os
import subprocess
import platform
import time

# Force the Agg backend for compatibility
matplotlib.use('Agg')

def plot(scores, mean_scores):
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    
    # Add text labels for current scores
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    # Save to a file
    filename = 'training_plot.png'
    plt.savefig(filename)
    plt.close()
    
    # Print progress information
    print("=" * 50)
    print(f"TRAINING PROGRESS")
    print("=" * 50)
    if len(scores) > 0:
        print(f"Game: {len(scores)}")
        print(f"Current Score: {scores[-1]}")
        print(f"Mean Score: {mean_scores[-1]:.2f}")
    print(f"Plot saved to: {os.path.abspath(filename)}")
    print("=" * 50)
    
    # Open the image file with the default system viewer
    # But do this only every 5 games to avoid opening too many windows
    if len(scores) % 5 == 0 or len(scores) == 1:
        try:
            # Determine the operating system and use appropriate command
            system = platform.system()
            filepath = os.path.abspath(filename)
            
            if system == 'Darwin':  # macOS
                subprocess.Popen(['open', filepath])
            elif system == 'Windows':
                os.startfile(filepath)
            else:  # Linux and other Unix-like
                subprocess.Popen(['xdg-open', filepath])
                
            # Small delay to allow the image viewer to open
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Could not open the image automatically: {e}")
            print("Please open the saved image manually.")