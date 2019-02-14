import matplotlib.pyplot as plt
import pathlib
import typing

class Plot:
    def __init__ (self, *, row_count:int, col_count:int, size:float) -> None:
        self.fig, self.axis_vv = plt.subplots(
            row_count,
            col_count,
            squeeze=False,
            figsize=(size*col_count,size*row_count),
        )

    def axis (self, row:int, col:int) -> typing.Any: # TODO: Real type
        return self.axis_vv[row][col]

    def savefig (self, plot_p:pathlib.Path, *, tight_layout_kwargs:typing.Dict[str,typing.Any]=dict()) -> None:
        self.fig.tight_layout(**tight_layout_kwargs)
        plot_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_p), bbox_inches='tight')
        print(f'wrote to file "{plot_p}"')
        # VERY important to do this -- otherwise your memory will slowly fill up!
        # Not sure which one is actually sufficient -- apparently none of them are, YAY!
        plt.clf()
        plt.cla()
        plt.close()
        plt.close(self.fig)
        #del fig
        #del axis_vv
