"""
Train a classification model for the CIFAR-10 dataset.
"""
import torch
import data_setup, model_builder, engine, utils


# Setup hyperparameters
NUM_EPOCHS = 11
BATCH_SIZE = 32
LEARNING_RATE = 0.01

if __name__ == '__main__':
    # Setup data directory
    data_dir = 'data'

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.data_loader(
        data_dir=data_dir,
        batch_size=BATCH_SIZE
    )

    # Create VGG16 model from model_builder.py
    model = model_builder.VGG16(num_classes=len(class_names))

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Start training from engine.py
    results = engine.train(model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )
    print(results)
    utils.plot_loss_curves(results)
