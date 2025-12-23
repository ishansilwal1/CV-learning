from ultralytics import YOLO, SAM

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

    # model = YOLO("yolov5n.pt")
    # model.info()

    # results = model.train(data="coco8.yaml", project = r"runs/custom_training", epochs=5, imgsz=640)

     # Load a model
    model = SAM("sam_b.pt")
    model.info()

    print("Starting inference...")
    results = model("D:\\CV practice\\Deep_learning\\dataset\\Train\\images\\_image_1763623303_aug_20251212-164529053914.jpg")
    print("Inference complete.")

    # Save each result image
    for i, res in enumerate(results):
        res.save(filename=f"output_{i}.jpg")
