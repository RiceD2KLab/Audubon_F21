from tkinter import *
import Training_only

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
    data_dir.insert(0, "./data")
    data_dir.grid(row=0, column=1, padx=10, pady=5)

    Label(train_frame, text='Image Extension').grid(row=1, column=0, padx=10, pady=5)
    img_ext = Entry(train_frame, width=40)
    img_ext.insert(0, ".JPEG")
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
    batch_size.insert(0, "8")
    batch_size.grid(row=6, column=1, padx=10, pady=5)

    Label(train_frame, text='Output Directory').grid(row=7, column=0, padx=10, pady=5)
    output_dir = Entry(train_frame, width=40)
    output_dir.insert(0, "./output")
    output_dir.grid(row=7, column=1, padx=10, pady=5)

    # Confusion matrix TODO

    Button(train_frame, text="Train Model", width=25, command=lambda: train_button_click([data_dir.get(),
                                                                                          img_ext.get(),
                                                                                          dir_ignore.get(),
                                                                                          num_workers.get(),
                                                                                          bayesian_iters.get(),
                                                                                          max_iters.get(),
                                                                                          batch_size.get(),
                                                                                          output_dir.get()]))\
        .grid(row=8)

    train_frame.grid(row=0)



def train_button_click(args):
    print(args)
    # Training_only.run([])
    pass


def run_mode_activate():
    initial_frame.pack_forget()
    initial_frame.destroy()

    Label(run_frame, text='Trained Model:').grid(row=0, column=0, padx=10)

    # Dropdown menu options
    options = [
        "Model1",
        "Model2"
    ]

    # datatype of menu text
    clicked = StringVar()

    # initial menu text
    clicked.set("Model1")

    # Create Dropdown menu
    drop = OptionMenu(run_frame, clicked, *options)
    drop.grid(row=0, column=1, padx=10)

    run_b = Button(run_frame, text="Run Trained Model", width=25, command=run_button_click)
    run_b.grid(row=0, column=2, padx=10)

    run_frame.grid(row=0)


def run_button_click():
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
