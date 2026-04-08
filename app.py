def show_menu() -> None:
    print("\n=== Simple Console App ===")
    print("1) Say hello")
    print("2) Add two numbers")
    print("3) Quit")


def say_hello() -> None:
    name = input("Enter your name: ").strip()
    if not name:
        name = "there"
    print(f"Hello, {name}!")


def add_two_numbers() -> None:
    try:
        first = float(input("First number: ").strip())
        second = float(input("Second number: ").strip())
    except ValueError:
        print("Please enter valid numbers.")
        return

    total = first + second
    print(f"Result: {first} + {second} = {total}")


def main() -> None:
    while True:
        show_menu()
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            say_hello()
        elif choice == "2":
            add_two_numbers()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
