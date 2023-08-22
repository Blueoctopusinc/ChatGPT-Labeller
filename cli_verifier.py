import argparse
import json
import curses

shape_categories_map = {
    1: "Saucer or Disk",
    2: "Triangle",
    3: "Cylinder/Cigar",
    4: "Sphere",
    5: "Oval",
    6: "Light",
    7: "Boomerang or V-shape",
    8: "Boxy",
    9: "No shape mentioned"
}

def main(stdscr, file_path):
    # Initialize colors for curses
    curses.start_color()
    for i in range(1, 10):
        curses.init_pair(i, curses.COLOR_BLACK, i)

    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get screen height and width
    height, width = stdscr.getmaxyx()

    # Scroll offset for the description
    scroll_offset = 0

    # Iterate over each item
    for key, item in data.items():
        stdscr.clear()

        # Split the description into lines
        description_lines = item["description"].split('\n')

        # Print "Description" label
        stdscr.addstr(0, 0, "Description:")

        # Print "Existing Labels" label
        stdscr.addstr(3, 0, "Existing Labels:")

        # Display all possible categories
        stdscr.addstr(15, 0, "All Categories:")
        for idx, (id, label) in enumerate(shape_categories_map.items()):
            stdscr.addstr(16 + idx, 0, f"{id}. {label}")

        while True:
            # Print the description below the label
            for i in range(min(height - 10, len(description_lines) - scroll_offset)):
                stdscr.addstr(1 + i, 0, description_lines[i + scroll_offset])

            # Print the existing labels below the label
            for idx, label in enumerate(item["labels"]):
                stdscr.addstr(4 + idx, 0, shape_categories_map[label], curses.color_pair(label))

            # Wait for user input
            key = stdscr.getch()

            # Scroll up
            if key == curses.KEY_UP and scroll_offset > 0:
                scroll_offset -= 1

            # Scroll down
            if key == curses.KEY_DOWN and scroll_offset < len(description_lines) - (height - 10):
                scroll_offset += 1

            # Break the loop on Enter key
            if key == 10:
                break

            stdscr.clear()

        # Display options
        stdscr.addstr(26, 0, "Choose categories (e.g., '1 2 3'): ")
        user_input = stdscr.getstr().decode('utf-8')

        new_labels = list(map(int, user_input.split()))

        # If label already exists in the list, remove it, otherwise add it
        for label in new_labels:
            if label in item["labels"]:
                item["labels"].remove(label)
            else:
                item["labels"].append(label)

        # Save back to the file continuously
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    stdscr.getch()  # Wait for user key press

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify labels for UFO sightings")
    parser.add_argument('file_path', help="Path to the JSON file")
    args = parser.parse_args()

    curses.wrapper(main, args.file_path)
