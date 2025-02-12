def yesno(query, opt_true=("y", "yes"), opt_false=("n", "no")):
    while True:
        answer = input(query + f" [{','.join(opt_true)}|{','.join(opt_false)}]").lower()
        if answer in opt_true:
            return True
        elif answer in opt_false:
            return False
