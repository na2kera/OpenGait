FROM paperist/texlive-ja:latest

# ieicej technical reports require the Times-compatible newtx fonts.
RUN tlmgr install newtx
