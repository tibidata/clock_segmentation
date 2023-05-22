from src import model

if __name__ == "__main__":
    segmenter = model.Segmenter(image_path='images/faliora.jpg')
    time = segmenter.segment()
    segmenter.show_detected_lines()

    print(time)
