
                --- HOW TO NAVIGATE AND USE THIS TEMPLATE ---
                      created by @teampro (Annik), 2020


THE DIFFERENT FILES/FOLDERS:

main.tex: 
The main document, where everything is nested together.

report.sty:
Main styling and commands of the report. Here are all packages used (add new \usepackage{} here, as this is added to/is the preamble). New commands can also be added here.

images:
All images (png) and pdf's are stored here as subfiles.

sections:
The different chapters are organized in different subfiles. THIS IS WHERE YOU WRITE woohoo:)

PS: You can double click on most parts of the pdf and be taken to the code:)


------------------------------------------------------------------------


WHEN WRITING:

Writing in LaTeX is a bit different, but not difficult to get used to. Everything except spaces and typing letters needs commands. 

comment --> In code, just use %.
            When commenting text, you can mark a word/section and a comment icon will appear.

tabs --> \quad or \qquad
         Ex. Some text here \quad and more text here

new line --> \\
             Ex.:Here is one line \\ 
             Now this text is on the next line
             (This is not considered "good practice", you could also just use \newline)
            
sections --> \section{eks: 4}  
             \subsection{eks: 4.1}
             \subsubsection{eks: 4.1.1}
             \paragraph{eks: 4.1.1.1}   %self-made
             
             The first one is already given to the different chapters. When you're adding them in the chapter.tex files, they'll be automatically included in the table of contents.

paragraphs --> \newpara
               This is a self-made command. See report.sty if you'd like to add commands or change it. 

adding images --> \includegraphics[width=? or hight=?]{images/Name_of_image}

todo-notes -->  \todo{what_you_want_to_say}
                These can be added to the text it concerns. They'll be automatically added to as a todo-list at the end of the report. 

code --> Ex1, write in latex:
         \begin{lstlisting}[language=C++, title=\textbf{{C++ code using listings}}]
                write code here
         \end{lstlisting}

         Ex2, input code as document:
         \lstinputlisting[language=Python]{source_filename.py}

         Easier if everybody uses these listings (see "metode" for detailed example and url below for details)



Everything is just one google search away, but here are some guides I things you might want to use:

% Change font --> https://www.overleaf.com/learn/latex/font_typefaces
% Text styling --> https://www.overleaf.com/learn/latex/Font_sizes,_families,_and_styles (also see how text is placed and styled on the first page as an example under "report.sty")
% Math --> https://en.wikibooks.org/wiki/LaTeX/Mathematics
% Code --> https://en.wikibooks.org/wiki/LaTeX/Source_Code_Listings
% Tables --> https://www.overleaf.com/learn/latex/tables
% Lists --> https://www.overleaf.com/learn/latex/lists


-----------------------------------------------------------------------


KEEP IN MIND:

Spell check --> menu in upper left corner and select language/off
    All comments are in English, therefore I would set it to that or turn it off. When you are writing your text in Norwegian, switch it so that it'll be checked.



