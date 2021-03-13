#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


print(f"this is {__name__}'s __init__ outside main.")


def main():
    print(f"this is {__name__}'s __init__ inside main.")


if __name__ == '__main__':
    main()

