from tkinter import *
import Training_only
import detect_only
import os

main_window = Tk()
main_window.geometry("500x300")
initial_frame = Frame(main_window)
train_frame = Frame(main_window)
run_frame = Frame(main_window)


def train_mode_activate():
    initial_frame.pack_forget()
    initial_frame.destroy()

    Label(train_frame, text='Data Directory').grid(row=0, column=0, padx=10, pady=5)
    data_dir = Entry(train_frame, width=40)
    data_dir.insert(0, 'C://Users\VelocityUser\Documents\D2K TDS D\TDS D-10')
    data_dir.grid(row=0, column=1, padx=10, pady=5)

    Label(train_frame, text='Image Extension').grid(row=1, column=0, padx=10, pady=5)
    img_ext = Entry(train_frame, width=40)
    img_ext.insert(0, ".JPG")
    img_ext.grid(row=1, column=1, padx=10, pady=5)

    Label(train_frame, text='Directory to Ignore').grid(row=2, column=0, padx=10, pady=5)
    dir_ignore = Entry(train_frame, width=40)
    dir_ignore.grid(row=2, column=1, padx=10, pady=5)

    Label(train_frame, text='Number of Workers').grid(row=3, column=0, padx=10, pady=5)
    num_workers = Entry(train_frame, width=40)
    num_workers.insert(0, "4")
    num_workers.grid(row=3, column=1, padx=10, pady=5)

    Label(train_frame, text='Max Iterations').grid(row=4, column=0, padx=10, pady=5)
    max_iters = Entry(train_frame, width=40)
    max_iters.insert(0, "500")
    max_iters.grid(row=4, column=1, padx=10, pady=5)

    Label(train_frame, text='Bayesian Tuning Iterations').grid(row=5, column=0, padx=10, pady=5)
    bayesian_iters = Entry(train_frame, width=40)
    bayesian_iters.insert(0, "40")
    bayesian_iters.grid(row=5, column=1, padx=10, pady=5)

    Label(train_frame, text='Batch Size').grid(row=6, column=0, padx=10, pady=5)
    batch_size = Entry(train_frame, width=40)
    batch_size.insert(0, "6")
    batch_size.grid(row=6, column=1, padx=10, pady=5)

    Label(train_frame, text='The Output Directory for the trained model is fixed as \"C://Users\\VelocityUser\\Documents\\Training_models\\\"')\
        .grid(row=7, padx=10, pady=5, columnspan=2)

    # Confusion matrix TODO

    Button(train_frame, text="Train Model", width=25, command=lambda: train_button_click([data_dir.get(),
                                                                                          img_ext.get(),
                                                                                          dir_ignore.get(),
                                                                                          num_workers.get(),
                                                                                          max_iters.get(),
                                                                                          bayesian_iters.get(),
                                                                                          batch_size.get(),
                                                                                          "C://Users\\VelocityUser\\Documents\\Training_models\\"]))\
        .grid(row=8)

    train_frame.grid(row=0)



def train_button_click(args):
    Training_only.run(args)
    pass


def run_mode_activate():
    initial_frame.pack_forget()
    initial_frame.destroy()

    model_dirs = []
    for file in os.listdir("C://Users\\VelocityUser\\Documents\\Training_models\\"):
        d = os.path.join("C://Users\\VelocityUser\\Documents\\Training_models\\", file)
        if os.path.isdir(d):
            model_dirs.append(d)

    Label(run_frame, text='Trained Model:').grid(row=0, column=0, padx=10)

    # Dropdown menu options
    options = model_dirs

    # datatype of menu text
    clicked = StringVar()

    # initial menu text
    clicked.set("Please select model from this dropdown list")

    # Create Dropdown menu
    drop = OptionMenu(run_frame, clicked, *options)
    drop.grid(row=0, column=1, padx=10)

    run_b = Button(run_frame, text="Run Trained Model", width=25, command=lambda: run_button_click([clicked.get()]))
    run_b.grid(row=0, column=2, padx=10)

    Label(run_frame, text='Loading images from Documents\\detect_images').grid(row=1, padx=10, columnspan=3)

    run_frame.grid(row=0)


def run_button_click(argv):
    detect_only.run(argv)
    pass


def create_initial_frame():
    ourMessage = "Welcome to Houston Audubon x D2K Lab at Rice University's collaborative project!\nOur project uses " \
                 "state of the art computer vision to train models that can automatically survey birds from overhead " \
                 "drone images.\n To train a new model, click \"train model\". To run an already trained model, click " \
                 "\"run model\"."

    messageVar = Label(initial_frame, text=ourMessage, relief=RAISED, font=("Arial", 12), wraplength=400, pady=10)
    messageVar.config(bg='lightgreen')
    messageVar.pack(fill="x")

    initial_frame.pack(fill="x")

    train_b = Button(initial_frame, text="Train Model", height=5, width=25, command=train_mode_activate)
    train_b.pack(side=LEFT, padx=25, pady=20)

    run_b = Button(initial_frame, text="Run Trained Model", height=5, width=25, command=run_mode_activate)
    run_b.pack(side=RIGHT, padx=25, pady=20)


create_initial_frame()


main_window.mainloop()
