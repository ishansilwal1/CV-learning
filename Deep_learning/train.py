from ultralytics import YOLO

if __name__ == '__main__':  
    # model = YOLO("yolo11n.pt") 
    # results = model.train(
    #     data=r"D:\CV practice\Deep_learning\dataset\data.yaml", 
    #     epochs=30,      
    #     imgsz=640,       
    #     batch=3,        
    #     device=0,        
    #     name="my_custom_model" 
    # )

    model = YOLO("yolov5n.pt")
    model.info()

    results = model.train(data="coco8.yaml", project = r"runs/custom_training", epochs=5, imgsz=640)